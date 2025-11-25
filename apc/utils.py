#!/usr/bin/env python3

from simple_einet.sampling_utils import sample_categorical_differentiably
import os
import json
import os
from torch.optim.lr_scheduler import OneCycleLR
import tqdm
import numpy as np
import torchvision
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from fast_pytorch_kmeans import KMeans
import umap
from colorlog import ColoredFormatter
import pytorch_warmup as warmup


from torchmetrics.clustering import (
    NormalizedMutualInfoScore,
    AdjustedMutualInfoScore,
    AdjustedRandScore,
    CompletenessScore,
    FowlkesMallowsIndex,
    HomogeneityScore,
    VMeasureScore,
)

from typing import Optional, Union
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn

from apc.models.abstract_model import AbstractAutoencoder
from apc.models.autoencoder.autoencoding_pc import APC
from apc.enums import PcEncoderType, ModelName, DecoderType
import os

from sklearn.manifold import TSNE
import torch
import wandb
from rtpt import RTPT

from apc.models.autoencoder.vanilla_autoencoder import VanillaAutoencoder
from apc.models.autoencoder.variational_autoencoder import VariationalAutoencoder
from apc.models.autoencoder.vaem import VAEM
from apc.models.autoencoder.miwae import MIWAE
from apc.models.autoencoder.hivae import HIVAE
from apc.models.iterative_imputer import IterativeImputer
from simple_einet.data import Dist, get_data_num_classes, is_1d_data

import logging

logger = logging.getLogger(__name__)


def log_images(images, tag, iteration, caption=""):
    """
    Log images to wandb.

    Args:
        images (torch.Tensor): Images to log.
        tag (str): Tag for wandb.
        iteration (int): Iteration for wandb.
        caption (str): Caption for wandb.
    """
    grid_recs = torchvision.utils.make_grid(
        images.float(),
        nrow=10,
        normalize=True,
        padding=1,
        pad_value=1.0,
    )
    # Save image in wandb
    wandb.log({tag: [wandb.Image(grid_recs, caption=caption)], "iteration": iteration}, commit=False)


def extract_classes_0_to_9(val_loader, dataset, N=1):
    # Sample data from val set for visual inspections
    collected_images = {i: [] for i in range(10)}
    for images, labels in val_loader:
        if any([s in dataset.lower() for s in ["celeba", "lsun"]]):
            return images[: 10 * N]
        for image, label in zip(images, labels):
            label = label.item()
            if label in collected_images and len(collected_images[label]) < N:
                collected_images[label].append(image)
            if all(len(collected_images[i]) == N for i in range(10)):
                break
        if all(len(collected_images[i]) == N for i in range(10)):
            break

    # Concatenate images in the specified order
    sorted_images = []
    for j in range(N):
        for i in range(10):
            sorted_images.append(collected_images[i][j])

    # Concatenate images into a single batch
    data_const = torch.stack(sorted_images, dim=0)
    return data_const


def interleave_tensors(tensor1, tensor2, chunk_size=10, rows=3):
    # Reshape tensors to have rows of chunk_size elements
    collection = []
    for i in range(rows):
        collection.append(tensor1[i * chunk_size : (i + 1) * chunk_size])
        collection.append(tensor2[i * chunk_size : (i + 1) * chunk_size])

    # Concatenate tensors
    interleaved = torch.cat(collection, dim=0)
    return interleaved


def auc(x: Union[list, np.ndarray, torch.Tensor], y: Union[list, np.ndarray, torch.Tensor]):
    """
    Compute the area under the curve using the trapezoidal rule (sklearn).

    Normalizes x to be in [0, 1].

    Args:
        x: x-coordinates.
        y: y-coordinates.

    Returns:
        float: Area under the curve.
    """

    if isinstance(x, list):
        x = np.array(x)

    if isinstance(y, list):
        y = np.array(y)

    if isinstance(x, torch.Tensor):
        x = x.numpy()

    if isinstance(y, torch.Tensor):
        y = y.numpy()

    # Normalize x to be in [0, 1]
    x = (x - x.min()) / (x.max() - x.min())
    return metrics.auc(x, y)


def _normalize(x, value=255.0):
    shape = x.shape
    x = x.view(shape[0], -1)
    x = x - x.min(-1, keepdim=True)[0]
    x = x / x.max(-1, keepdim=True)[0] * value
    return x.view(shape)


def get_val_iterations(iterations_total: int, num_val: int) -> list[int]:
    """
    Get iteration points for evaluation.

    Args:
        iterations_total (int): Total number of iterations.
        num_val (int): Number of evaluation points.

    Returns:
        List[int]: List of iterations to evaluate on.
    """
    iters = [round(x) for x in np.linspace(0, iterations_total - 1, num_val + 1, endpoint=True)]
    return iters[1:]  # Drop first iteration at step 0


@torch.no_grad()
def make_images(
    model: AbstractAutoencoder,
    data_const: torch.Tensor,
    iteration: int,
    seed: int,
    cfg: DictConfig,
    ps: list[int] = None,
    results_dir: str | None = None,
    save_images: bool = False,
    prefix: str = "",
):
    """
    Make images from model and log them to wandb.

    Args:
        model: Model to use for visualizations.
        data_const: Data to use for visualizations.
        iteration: Iteration for wandb.
        seed: Seed for random number generator.
    """
    if ps is None:
        ps = [50, 90]

    vis_dir = os.path.join(results_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    data_shape = data_const.shape[1:]
    data_const = data_const.float()

    # Reconstruction of data
    for dt in model.decoder_types():
        # encoding = model.encode(data_const, mpe=True)
        # x_rec = model.decode(encoding, decoder_type=dt)
        x_rec = model.reconstruct(data_const, decoder_type=dt)
        log_images(
            interleave_tensors(data_const, x_rec, chunk_size=10),
            f"{prefix}vis/rec/{dt}/full",
            iteration=iteration,
            caption="Top: data, bottom: reconstruction",
        )

    taus = [0.5, 1.0]

    for tau in taus:
        # Sample z ~ p(Z)
        z_sample = model.sample_z(num_samples=data_const.shape[0] * 2, seed=seed, tau=tau)

        # Decode
        for dt in model.decoder_types():
            x_sample = model.decode(z_sample, decoder_type=dt)
            log_images(x_sample, f"{prefix}vis/{dt}/g(z ~ p(Z)), tau={tau}", iteration=iteration)

    # Sample PC x ~ p(X)
    if DecoderType.PC in model.decoder_types():
        for tau in taus:
            x_sample = model.sample_x(
                num_samples=data_const.shape[0] * 2, seed=seed, decoder_type=DecoderType.PC, tau=tau
            )
            log_images(x_sample, f"{prefix}vis/{DecoderType.PC}/x ~ p(X), tau={tau}", iteration=iteration)
            if save_images:
                torch.save(x_sample.cpu(), os.path.join(vis_dir, f"x_sample_tau_{tau}.pt"))

    # Hold out p % of the data features randomly and reconstruct
    for p in ps:

        linspace = torch.linspace(0, 1, data_const.shape[2] * data_const.shape[3], device=data_const.device)
        rand = torch.rand(data_const.shape[0], linspace.shape[0], device=data_const.device)
        perm_indices = torch.argsort(rand, dim=1)
        rand = linspace[perm_indices].view(data_const.shape[0], 1, data_const.shape[2], data_const.shape[3])
        mask = rand < (p / 100)

        # Repeat mask at dim=1 exactly data_shape[1] times to ensure that the full pixel is selected
        mask = mask.repeat(1, data_shape[0], 1, 1)
        for dt in model.decoder_types():
            # Create new data tensor (clone)
            data_cloned = data_const.clone()
            data_cloned[mask] = torch.tensor(float("nan"))

            recs = model.reconstruct(data_cloned, decoder_type=dt, fill_evidence=False)

            # Log reconstructions
            data_cloned = data_const.clone()
            data_cloned[mask] = cfg.nan_mask_value
            log_images(
                interleave_tensors(data_cloned, recs.view(-1, *data_shape), chunk_size=10),
                f"{prefix}vis/rec/{dt}/rand-{p}",
                iteration=iteration,
            )
            # Save tensor (not image) to results_dir/vis
            if save_images:
                torch.save(recs.cpu(), os.path.join(vis_dir, f"recs_rand_{p}.pt"))
                data_cloned[mask] = float("nan")
                torch.save(data_cloned.cpu(), os.path.join(vis_dir, f"data_masked_rand_{p}.pt"))
                torch.save(data_const.cpu(), os.path.join(vis_dir, f"data_const.pt"))

    # Hold out lower half of the images
    mask = torch.zeros_like(data_const).bool()
    mask[..., data_shape[1] // 2 :, :] = True

    data_cloned = data_const.clone()
    data_cloned[mask] = torch.tensor(float("nan"))
    for dt in model.decoder_types():
        # Create new data tensor (clone)
        data_cloned = data_const.clone()
        data_cloned[mask] = torch.tensor(float("nan"))
        recs = model.reconstruct(data_cloned, decoder_type=dt)

        # Log reconstructions
        data_cloned = data_cloned.clone()
        data_cloned[mask] = 0.0
        log_images(
            interleave_tensors(data_cloned, recs.view(-1, *data_shape), chunk_size=10),
            f"{prefix}vis/rec/{dt}/lower",
            iteration=iteration,
        )

    # Hold out right half of the images
    mask = torch.zeros_like(data_const).bool()
    mask[..., :, data_shape[1] // 2 :] = True

    for dt in model.decoder_types():
        # Create new data tensor (clone)
        data_cloned = data_const.clone()
        data_cloned[mask] = torch.tensor(float("nan"))
        recs = model.reconstruct(data_cloned, decoder_type=dt)

        # Log reconstructions
        data_cloned = data_const.clone()
        data_cloned[mask] = 0.0
        log_images(
            interleave_tensors(data_cloned, recs.view(-1, *data_shape), chunk_size=10),
            f"{prefix}vis/rec/{dt}/right",
            iteration=iteration,
        )

    # Hold out center square 20% of the image
    mask = torch.zeros_like(data_const).bool()
    mask[
        ...,
        data_shape[1] // 2 - data_shape[1] // 5 : data_shape[1] // 2 + data_shape[1] // 5,
        data_shape[1] // 2 - data_shape[1] // 5 : data_shape[1] // 2 + data_shape[1] // 5,
    ] = True

    for dt in model.decoder_types():
        # Create new data tensor (clone)
        data_cloned = data_const.clone()
        data_cloned[mask] = torch.tensor(float("nan"))
        recs = model.reconstruct(data_cloned, decoder_type=dt)

        # Log reconstructions
        data_cloned = data_const.clone()
        data_cloned[mask] = 0.0
        log_images(
            interleave_tensors(data_cloned, recs.view(-1, *data_shape), chunk_size=10),
            f"{prefix}vis/rec/{dt}/center",
            iteration=iteration,
        )


def setup_rtpt(cfg: DictConfig, iters: int, add_tag: str = ""):
    """
    Setup RTPT based on the given config.

    Args:
        cfg: Hydra config object.

    Returns:
        RTPT: RTPT object.
    """

    return RTPT(
        name_initials="SB",
        experiment_name=cfg.dataset + "_" + (cfg.tag if cfg.tag else "") + add_tag,
        max_iterations=iters,
        update_interval=cfg.log_interval * 5,
    )


def preprocess_cfg(cfg: DictConfig):
    """
    Preprocesses the config file.
    Replace defaults if not set (such as data/results dir).

    Args:
        cfg: Config file.
    """
    home = os.getenv("HOME")

    # If results dir is not set, get from ENV, else take ~/data
    if "data_dir" not in cfg:
        cfg.data_dir = os.getenv("DATA_DIR", default=os.path.join(home, "data"))

    # If results dir is not set, get from ENV, else take ~/results
    if "results_dir" not in cfg:
        cfg.results_dir = os.getenv("RESULTS_DIR", default=os.path.join(home, "results"))

    # If FP16/FP32 is given, convert to int (else it's "bf16", keep string)
    if cfg.precision == "16" or cfg.precision == "32":
        cfg.precision = int(cfg.precision)

    if "profiler" not in cfg:
        cfg.profiler = None  # Accepted by PyTorch Lightning Trainer class

    if "tag" not in cfg:
        cfg.tag = None

    if "group" not in cfg:
        cfg.group = None

    # Convert dist string to enum
    cfg.einet.dist = Dist[cfg.einet.dist.upper()]
    cfg.apc.encoder = PcEncoderType[cfg.apc.encoder.upper()]
    cfg.model_name = ModelName[cfg.model_name.upper()]


def human_readable_number(number: int):
    """
    Convert a number to a human-readable format.

    Args:
        number: Number to convert.

    Returns:
        str: Human readable number.
    """
    suffixes = ["", "K", "M", "B", "T"]
    suffix_index = 0

    while number >= 1000 and suffix_index < len(suffixes) - 1:
        number /= 1000
        suffix_index += 1

    return f"{number:.2f}" + suffixes[suffix_index]


def count_params(cfg_container: dict, model: nn.Module):
    """
    Count number of parameters in the model. Prints the count and store it in the given config container.

    Args:
        cfg_container: Config container.
        model: Model.
    """

    num_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_encoder = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    num_params_decoder = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    cfg_container["num_params_total"] = num_params_total
    cfg_container["num_params_encoder"] = num_params_encoder
    cfg_container["num_params_decoder"] = num_params_decoder
    logger.info("Number of parameters Total:   " + human_readable_number(num_params_total))
    logger.info("Number of parameters Encoder: " + human_readable_number(num_params_encoder))
    logger.info("Number of parameters Decoder: " + human_readable_number(num_params_decoder))


def make_optimizer(optimizer_name: str, optim_kwargs: dict):
    if optimizer_name == "adam":
        return torch.optim.Adam(**optim_kwargs)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(**optim_kwargs)
    elif optimizer_name == "sgd":
        if "amsgrad" in optim_kwargs:
            amsgrad = optim_kwargs.pop("amsgrad")
        return torch.optim.SGD(**optim_kwargs)
    elif optimizer_name == "rmsprop":
        if "amsgrad" in optim_kwargs:
            amsgrad = optim_kwargs.pop("amsgrad")
        return torch.optim.RMSprop(**optim_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def setup_model(cfg):
    """
    Create model, optimizer and lr_scheduler based on the given config.

    Args:
        cfg: Hydra config object.

    Returns:
        lr_scheduler: Learning rate scheduler.
        model: Model.
        optimizer: Optimizer.
    """
    match cfg.model_name:
        case ModelName.APC:
            model = APC(cfg)
        case ModelName.AE:
            model = VanillaAutoencoder(cfg)
        case ModelName.VAE:
            model = VariationalAutoencoder(cfg)
        case ModelName.VAEM:
            model = VAEM(cfg)
        case ModelName.MIWAE:
            model = MIWAE(cfg)
        case ModelName.HIVAE:
            model = HIVAE(cfg)
        case ModelName.MICE | ModelName.MISSFOREST:
            model = IterativeImputer(cfg)
        case _:
            raise ValueError(f"Model_name {cfg.model_name} not supported")
    # Initialize the Adam optimizer with the model parameters and learning rate specified in the configuration

    params = [
        {"params": model.encoder.parameters(), "lr": cfg.train.lr_encoder},
        {"params": model.decoder.parameters(), "lr": cfg.train.lr_decoder},
    ]
    optimizer = make_optimizer(
        optimizer_name=cfg.train.optim,
        optim_kwargs={
            "params": params,
            "weight_decay": cfg.train.weight_decay,
            "amsgrad": cfg.train.amsgrad,
        },
    )

    # Initialize the learning rate scheduler with the optimizer and milestones specified in the configuration
    if cfg.train.lr_scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(0.66 * cfg.train.iters), int(0.9 * cfg.train.iters)], gamma=1e-1
        )

    elif cfg.train.lr_scheduler == "onecycle":
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg.train.lr,
            total_steps=cfg.train.iters + 1,  # +1 b/c 1cycle has a bug in its last step where it upticks the lr again
            div_factor=25,
            final_div_factor=1e4,
        )
    elif cfg.train.lr_scheduler == "plateau":
        patience_factor = 0.05  # 5% of total iterations
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=int(patience_factor * cfg.train.iters),
            min_lr=cfg.train.lr / 1000,
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {cfg.train.lr_scheduler}")

    # Warmup scheduler
    if cfg.train.warmup_enabled:
        warmup_scheduler = warmup.ExponentialWarmup(
            optimizer, warmup_period=int(cfg.train.iters * cfg.train.warmup_p / 100)
        )
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler, model, optimizer


def determine_normalization(cfg: DictConfig) -> bool:
    """Determine whether to normalize the data based on the configuration."""
    if (
        cfg.model_name == ModelName.APC
        and cfg.apc.encoder in (PcEncoderType.EINET, PcEncoderType.EINET_ACT, PcEncoderType.EINET_CAT)
        and (cfg.einet.dist in (Dist.BINOMIAL, Dist.BERNOULLI, Dist.CATEGORICAL))
    ):
        return False
    elif cfg.model_name == ModelName.APC and cfg.apc.encoder in (
        PcEncoderType.CONV_PC,
        PcEncoderType.CONV_PC_SPNAE_ACT,
        PcEncoderType.CONV_PC_SPNAE_CAT,
    ):
        if cfg.conv_pc.dist_data in (Dist.BINOMIAL, Dist.BERNOULLI, Dist.CATEGORICAL):
            return False
        elif cfg.conv_pc.dist_data in (Dist.NORMAL, Dist.NORMAL_RAT, Dist.NORMAL_MEAN, Dist.LAPLACE):
            return True
        else:
            raise ValueError(f"Unknown dist_data: {cfg.conv_pc.dist_data}")
    elif is_1d_data(cfg.dataset):
        return True
    else:
        return True


def setup_logging(dir, name: Optional[str] = None):
    """
    Sets up logging with both stdout and file handlers, using colorful output for stdout.

    Args:
        log_file_directory (str): The directory where the log file will be stored.

    Raises:
        ValueError: If the provided directory does not exist and cannot be created.
    """
    # Ensure the directory exists
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            raise ValueError(f"Failed to create log directory: {e}")

    # Define log file path
    name = name if name is not None else "logger"
    log_file_path = os.path.join(dir, name)

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level to DEBUG for detailed logs

    # Create stdout handler
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)  # Set to INFO level for console output

    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)  # Set to DEBUG level for file output

    # Create a formatter for the file handler
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    color_formatter = ColoredFormatter(
        fmt="%(asctime_log_color)s%(asctime)s %(name_log_color)s[%(name)s] "
        "%(levelname_log_color)s[%(levelname)s] %(message_log_color)s- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={
            "asctime": {"DEBUG": "blue", "INFO": "blue", "WARNING": "blue", "ERROR": "red", "CRITICAL": "red"},
            "name": {"DEBUG": "cyan", "INFO": "cyan", "WARNING": "cyan", "ERROR": "cyan", "CRITICAL": "cyan"},
            "levelname": {"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red"},
            "message": {"DEBUG": "white", "INFO": "white", "WARNING": "white", "ERROR": "white", "CRITICAL": "white"},
        },
    )

    # Attach the formatters to the handlers
    stdout_handler.setFormatter(color_formatter)
    file_handler.setFormatter(file_formatter)

    # Add the handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    logging.info("Logging is set up with stdout and file handlers.")


def postprocess_latents(z, cfg):
    # if cfg.model_name == "apc" and cfg.apc.encoder == ApcEncoderType.CONV_PC_SPNAE_CAT:
    #     z = z.max(dim=1).indices.view(z.shape[0], cfg.latent_dim).float()
    z = z.view(z.shape[0], cfg.latent_dim)
    return z
