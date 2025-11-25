import json
from apc.enums import ModelName, PcEncoderType
import os

import numpy as np
import seaborn as sns
import torch
import tqdm
import umap
import wandb
from fast_pytorch_kmeans import KMeans
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from torchmetrics.clustering import (
    NormalizedMutualInfoScore,
    AdjustedRandScore,
    AdjustedMutualInfoScore,
    CompletenessScore,
    HomogeneityScore,
    VMeasureScore,
    FowlkesMallowsIndex,
)

from apc.models.abstract_model import AbstractAutoencoder
from simple_einet.data import get_data_num_classes
from apc.utils import postprocess_latents


@torch.no_grad()
def make_vis_embedding(
    model: AbstractAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    iteration: int | None = None,
    seed: int | None = None,
    cfg: DictConfig | None = None,
    function: str = "tsne",
    results_dir: str = None,
    prefix: str = "",
    ps: list[int] | None = None,
):
    assert function in ["tsne", "umap"]

    # Manual generator
    generator = torch.Generator()

    if cfg.test.tsne_all:
        if ps is None:
            ps = list(range(0, 100, 10)) if not cfg.debug else [0, 10]
    else:
        ps = [0]

    results = {}

    # Define scores
    scores = [
        ("nmi", "Normalized Mutual Information", NormalizedMutualInfoScore),
        ("ari", "Adjusted Rand", AdjustedRandScore),
        ("ami", "Adjusted Mutual Information", AdjustedMutualInfoScore),
        ("completeness", "Completeness", CompletenessScore),
        ("homogeneity", "Homogeneity", HomogeneityScore),
        ("v-measure", "V-Measure", VMeasureScore),
        ("fmi", "Fowlkes-Mallows Index", FowlkesMallowsIndex),
    ]

    for p in tqdm.tqdm(ps, desc=f"{function.upper()} Embeddings"):

        # Sample data and their labels from val set for visual inspections
        # Assuming dataloader is your DataLoader instance
        collected_embeddings = []
        collected_labels = []
        count = 0

        for i, (images, labels) in enumerate(dataloader):
            images = images.float()

            # Use the same random tensor (same seed) as base for the mask such that p1 < p2 means mask1 is a subset of mask2
            generator.manual_seed(i)
            rand = torch.rand(images.shape[0], 1, images.shape[2], images.shape[3], generator=generator)
            mask = rand < p / 100

            # Repeat mask at dim=1 exactly data.shape[1] times to ensure that the full pixel is selected
            mask = mask.repeat(1, images.shape[1], 1, 1)

            images[mask] = torch.tensor(float("nan"))
            embeddings = model.encode(images.view(images.shape[0], images.shape[1], -1), mpe=True)
            embeddings = postprocess_latents(embeddings, cfg)

            collected_embeddings.append(embeddings)
            collected_labels.append(labels.cpu())
            count += images.shape[0]

            if count > 5000 or cfg.debug:
                break

        # Concatenate images into a single batch X
        X = torch.cat(collected_embeddings, dim=0).cpu()

        if cfg.model_name == ModelName.APC and cfg.apc.encoder == PcEncoderType.CONV_PC_SPNAE_ACT and not cfg.debug:
            # Only necessary for ACT_PC
            # Compute mean and std of X for normalization
            X_mean = X.mean(dim=0, keepdim=True)
            X_std = X.std(dim=0, keepdim=True)

            # Normalize X
            X = (X - X_mean) / X_std

        X = X.view(X.shape[0], -1).numpy()
        Y = torch.cat(collected_labels).cpu().numpy()
        if X.shape[0] < 30:
            perplexity = X.shape[0] // 3
        else:
            perplexity = 30  # TSNE default

        if function == "umap":
            X_embedded = umap.UMAP(
                n_neighbors=cfg.umap.n_neighbors, min_dist=cfg.umap.min_dist, n_jobs=min(os.cpu_count(), 16)
            ).fit_transform(X)
        else:
            X_embedded = TSNE(
                n_components=2, learning_rate="auto", init="random", random_state=seed, perplexity=perplexity
            ).fit_transform(X)

        # Save embeddings in results_dir
        assert results_dir is not None
        np.save(os.path.join(results_dir, f"{prefix.replace('/', '-')}{function}_embeddings_{p:0>2d}.npy"), X_embedded)
        np.save(
            os.path.join(results_dir, f"{prefix.replace('/', '-')}{function}_labels.npy"), Y
        )  # This will be the same for all p

        # Visualize the 2D embedding with matplotlib and seaborn
        sns.set(style="whitegrid")

        fig = plt.figure()
        sns.scatterplot(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            hue=Y,
            palette=sns.color_palette("deep", 10),
            # legend="full",
            alpha=1.0,
        )
        wandb.log(
            {f"{prefix}embeddings/{function}/vis/{p:0>2d}": wandb.Image(fig), "iteration": iteration}, commit=False
        )
