#!/usr/bin/env python3
from contextlib import nullcontext
import pytorch_warmup as warmup
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import tqdm
import wandb
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from apc.models.abstract_model import AbstractAutoencoder
from apc.enums import PcEncoderType, ModelName
from apc.models.encoder.nn_encoder import NeuralEncoder2D
from simple_einet.data import get_data_num_classes, is_1d_data, _make_loader
from apc.utils import auc, postprocess_latents
from apc.utils import get_val_iterations, make_optimizer

logger = logging.getLogger(__name__)


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        """
        Args:
            input_dim (int): Input dimension.
            hidden_dims (list[int]): Hidden dimension.
            output_dim (int): Output dimension.
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.layers = torch.nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
            else:
                self.layers.append(torch.nn.Linear(hidden_dims[i - 1], hidden_dim))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
        """
        super(LogisticRegression, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x


def precompute_embeddings(model_encoder: AbstractAutoencoder, dataloader: DataLoader, cfg: DictConfig):
    """Precompute embeddings for a given dataloader."""
    model_encoder.train()  # NOTE: eval() causes issues with APC
    embeddings = []
    targets = []

    with torch.no_grad():
        for data, target in tqdm.tqdm(dataloader, desc="Computing embeddings"):
            z = model_encoder.encode(data)
            z = postprocess_latents(z, cfg)
            embeddings.append(z.cpu())
            targets.append(target.cpu())

            if cfg.debug:
                break

    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)

    # Create TensorDataset with transformation
    return TensorDataset(embeddings, targets)


def test_downstream_task(
    model_encoder: AbstractAutoencoder,
    train_loader,
    val_loader,
    test_loader,
    cfg: DictConfig,
    rtpt,
    fabric,
    results_dir: str,
    prefix: str = "",
):
    """
    Train the PC model.

    Args:
        model_encoder (AbstractLvPc): Encoder model.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.

    """

    train_embeddings_dataset = precompute_embeddings(model_encoder, train_loader, cfg)

    if cfg.model_name in ModelName.APC and cfg.apc.encoder == PcEncoderType.CONV_PC_SPNAE_ACT and not cfg.debug:
        # Only necessary for ACT_PC
        # Compute mean and std of embeddings for normalization
        train_embeddings_mean = train_embeddings_dataset.tensors[0].mean(dim=0, keepdim=True)
        train_embeddings_std = train_embeddings_dataset.tensors[0].std(dim=0, keepdim=True)

        # Normalize embeddings
        train_embeddings_dataset.tensors = (
            (train_embeddings_dataset.tensors[0] - train_embeddings_mean) / train_embeddings_std,
            train_embeddings_dataset.tensors[1],
        )
    else:
        # Set to (0,1) for non-act models
        train_embeddings_mean = torch.tensor(0.0)
        train_embeddings_std = torch.tensor(1.0)

    train_embeddings_mean = fabric.to_device(train_embeddings_mean)
    train_embeddings_std = fabric.to_device(train_embeddings_std)

    MAX_WORKERS = min(os.cpu_count(), 8)
    if cfg.debug:
        MAX_WORKERS = 0

    # Create new dataloaders with embeddings
    train_loader_embeddings = _make_loader(
        dataset=train_embeddings_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        loop=True,
        seed=cfg.seed,
        num_workers=MAX_WORKERS,
    )

    # Fabric setup
    train_loader_embeddings = fabric.setup_dataloaders(train_loader_embeddings)

    eval_iterations = get_val_iterations(iterations_total=cfg.downstream_task.train.iters, num_val=cfg.num_val)

    num_classes = get_data_num_classes(cfg.dataset)

    for downstream_task_model in cfg.downstream_task.model:
        logger.info(f"Training downstream task model: {downstream_task_model}")
        # Construct downstream task model
        if downstream_task_model == "mlp":
            model_downstream = MLP(
                input_dim=cfg.latent_dim,
                hidden_dims=cfg.downstream_task.mlp.hidden_dims,
                output_dim=num_classes,
            )
        elif downstream_task_model == "lr":
            model_downstream = LogisticRegression(input_dim=cfg.latent_dim, output_dim=num_classes)
        else:
            raise ValueError(f"Unknown downstream task model: {downstream_task_model}")
        logger.info("\n" + str(model_downstream))

        # Setup optimizer for downstream task model
        optimizer = make_optimizer(
            optimizer_name=cfg.downstream_task.train.optim,
            optim_kwargs={
                "params": model_downstream.parameters(),
                "lr": cfg.downstream_task.train.lr,
                "weight_decay": cfg.downstream_task.train.weight_decay,
                "amsgrad": cfg.downstream_task.train.amsgrad,
            },
        )

        # Setup downstream model with fabric
        model_downstream, optimizer = fabric.setup(model_downstream, optimizer)

        # Setup learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.66 * cfg.downstream_task.train.iters), int(0.9 * cfg.downstream_task.train.iters)],
            gamma=1e-1,
        )

        # Warmup scheduler
        if cfg.downstream_task.train.warmup_enabled:
            warmup_scheduler = warmup.ExponentialWarmup(
                optimizer, warmup_period=int(cfg.downstream_task.train.iters * cfg.downstream_task.train.warmup_p / 100)
            )

        criterion = torch.nn.CrossEntropyLoss()

        pbar = tqdm.tqdm(train_loader_embeddings, total=cfg.downstream_task.train.iters, leave=False)
        for iteration, (z, target) in enumerate(pbar):
            if iteration == cfg.downstream_task.train.iters:
                break

            # Stop after a few batches in debug mode
            if cfg.debug and iteration > 2:
                break

            # Reset gradients
            optimizer.zero_grad()

            # Predict using downstream task model
            logits = model_downstream(z)

            # Compute loss
            loss = criterion(logits, target)

            # Compute gradients
            fabric.backward(loss)

            optimizer.step()

            # If warmup is enabled, use warmup dampening context, else use nullcontext
            if cfg.downstream_task.train.warmup_enabled:
                warmup_context = warmup_scheduler.dampening
            else:
                warmup_context = nullcontext

            # Optional warmup context
            with warmup_context():
                lr_scheduler.step()

            rtpt.step()

            # Logging
            if iteration % cfg.log_interval == 0:
                # Rewrite set_description with dict generator and string interpolation
                pbar.set_description("Loss: {:.4f}".format(loss.item()))

                # Log wandb
                wandb.log(
                    {
                        f"{prefix}train_loss/downstream/{downstream_task_model}/cross-entropy": loss.item(),
                        "iteration": iteration,
                    },
                    commit=False,
                )

                # Commit logs
                if iteration not in eval_iterations:
                    # If this is an evaluation iteration, the logs are commited at a later point anyway
                    wandb.log(data={}, commit=True)

            if iteration in eval_iterations or cfg.debug:
                # Commit logs
                wandb.log(data={}, commit=True)

        evaluate_downstream_task(
            model_encoder,
            model_downstream,
            test_loader,
            "test",
            iteration,
            debug=cfg.debug,
            cfg=cfg,
            results_dir=results_dir,
            train_embeddings_mean=train_embeddings_mean,
            train_embeddings_std=train_embeddings_std,
            downstream_task_model_tag=downstream_task_model,
            prefix=prefix,
        )


@torch.no_grad()
def evaluate_downstream_task(
    model_encoder,
    model_downstream,
    dataloader,
    tag,
    iteration=None,
    debug=False,
    cfg=None,
    results_dir: str = None,
    train_embeddings_mean=None,
    train_embeddings_std=None,
    downstream_task_model_tag=None,
    prefix: str = "",
):
    """
    Evaluate the downstream task model.

    Args:
        model_downstream (torch.nn.Module): Downstream task model.
        dataloader (torch.utils.data.DataLoader): Data loader.
        tag (str): Tag.
        iteration (int): Iteration.
        debug (bool): Debug mode.

    """
    model_encoder.eval()
    model_downstream.eval()

    total = 0

    # Init dict to store test mse
    metrics = defaultdict(float)

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def collect(data, target, key):

        # AE/VAE cannot marginalize -> set nan to 0.0 instead
        if not cfg.model_name in [ModelName.APC]:
            data[data.isnan()] = 0.0

        # Compute embeddings
        z = model_encoder.encode(data)

        z = postprocess_latents(z, cfg)

        # Normalize embeddings
        z = (z - train_embeddings_mean) / train_embeddings_std

        logits = model_downstream(z)

        assert not z.isnan().any(), "NaN in z"
        assert not logits.isnan().any(), "NaN in logits"

        # Cross-entropy
        loss = criterion(logits, target)
        metrics[f"{prefix}{tag}/downstream/{downstream_task_model_tag}/ce/{key}"] += loss.item()

        # Accuracy
        _, predicted = logits.max(1)
        metrics[f"{prefix}{tag}/downstream/{downstream_task_model_tag}/acc/{key}"] += predicted.eq(target).sum().item()

    # Percentages of holdout
    ps = list(np.arange(0, 100, 5))

    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    for i, (data, target) in enumerate(pbar):
        data = data.float()
        total += target.size(0)

        collect(data, target, "full")

        if not is_1d_data(cfg.dataset):  # Don't do this for 1D data
            #############################
            # Right half reconstruction #
            #############################
            # Hold out right half of the images
            mask = torch.zeros_like(data).bool()
            mask[..., data.shape[3] // 2 :] = True

            data_cloned = data.clone()
            data_cloned[mask] = torch.tensor(float("nan"))
            collect(data_cloned, target, "right")

            #############################
            # Lower half reconstruction #
            #############################
            mask = torch.zeros_like(data).bool()
            mask[..., data.shape[2] // 2 :, :] = True

            data_cloned = data.clone()
            data_cloned[mask] = torch.tensor(float("nan"))
            collect(data_cloned, target, "lower")

        #####################
        # p % Reconstuction #
        #####################

        # Use the same random tensor as base for the mask such that p1 < p2 means mask1 is a subset of mask2
        gen = torch.Generator(device=data.device)
        gen.manual_seed(cfg.seed + i)

        # Use the same random tensor as base for the mask such that p1 < p2 means mask1 is a subset of mask2
        linspace = torch.linspace(0, 1, data.shape[2] * data.shape[3], device=data.device)
        rand = torch.rand(data.shape[0], linspace.shape[0], device=data.device, generator=gen)
        perm_indices = torch.argsort(rand, dim=1)
        rand = linspace[perm_indices].view(data.shape[0], 1, data.shape[2], data.shape[3])

        for p in ps:

            ################
            # Missing data #
            ################
            mask = rand < (p / 100)
            # Repeat mask at dim=1 exactly data.shape[1] times to ensure that the full pixel is selected
            mask = mask.repeat(1, data.shape[1], 1, 1)
            data_cloned = data.clone()
            data_cloned[mask] = torch.tensor(float("nan"))
            collect(data_cloned, target, f"missing_{p}")

        if debug:
            break

    # Normalize metrics
    metrics = {k: v / total for k, v in metrics.items()}
    metrics["iteration"] = iteration
    # wandb.log(metrics, commit=False)

    if tag == "test":
        # Accuracy plot missing data
        accs = [metrics[f"{prefix}{tag}/downstream/{downstream_task_model_tag}/acc/missing_{p}"] for p in ps]
        data = [[x, y] for (x, y) in zip(ps, accs)]
        table = wandb.Table(data=data, columns=["percentage", "accuracy"])
        wandb_dict = {
            f"{prefix}accuracy_plot_missing_downstream_{downstream_task_model_tag}": wandb.plot.line(
                table,
                "percentage",
                "accuracy",
                stroke=None,
                title=f"Downstream Accuracy with missing Data ({downstream_task_model_tag.upper()})",
            )
        }

        # Add Area und Curve for missing data plot
        auc_downstream_acc = auc(ps, accs)
        metrics["auc-missing"] = auc_downstream_acc
        wandb_dict.update({f"{prefix}{tag}/downstream/{downstream_task_model_tag}/auc_missing": auc_downstream_acc})

        if not is_1d_data(cfg.dataset):  # Skip this for 1D data
            # Partial image holdout bar plot
            keys = ["lower", "right", "full"]
            accs = [metrics[f"{prefix}{tag}/downstream/{downstream_task_model_tag}/acc/{k}"] for k in keys]

            data = [[name, value] for (name, value) in zip(keys, accs)]
            table = wandb.Table(data=data, columns=["part", "accuracy"])
            wandb_dict.update(
                {
                    f"{prefix}holdout_plot_partial_image_downstream_accuracy_{downstream_task_model_tag}": wandb.plot.bar(
                        table,
                        "part",
                        "accuracy",
                        title=f"Downstream Accuracy with Partial Image ({downstream_task_model_tag.upper()})",
                    )
                }
            )

        wandb.log(wandb_dict, commit=True)

    # Save as json
    with open(
        os.path.join(
            results_dir, f"{prefix.replace('/', '-')}{tag}-downstream-{downstream_task_model_tag}-metrics.json"
        ),
        "w",
    ) as f:
        json.dump(metrics, f)

    model_downstream.train()
