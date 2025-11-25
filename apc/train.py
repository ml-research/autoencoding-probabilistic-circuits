from typing import Union


from contextlib import nullcontext
from apc.test import validation_loss
from apc.models.abstract_model import AbstractAutoencoder
from apc.models.iterative_imputer import IterativeImputer

from simple_einet.data import is_1d_data
import torch
import tqdm
import wandb

from utils import (
    get_val_iterations,
    make_images,
)
from early_stopping import EarlyStopping

import logging

logger = logging.getLogger(__name__)

def train_iterative_imputer(cfg, model: IterativeImputer, train_loader, val_loader, data_const, results_dir):
    # Collect cfg.iterative_imputer.n_train number of samples from the training set
    data = []
    n_samples = 0
    for i, (x, y) in enumerate(train_loader):
        data.append(x)
        n_samples += x.shape[0]
        if n_samples >= cfg.iterative_imputer.n_train:
            break
    data = torch.cat(data, dim=0)[:cfg.iterative_imputer.n_train]

    # Set the imputer data
    logger.info("Fitting imputer ...")
    model.fit(data)
    logger.info("Done fitting imputer ...")


    # Add images one last time
    is_image_data = not is_1d_data(cfg.dataset)
    if is_image_data:
        make_images(
            model,
            data_const,
            iteration=cfg.train.iters,
            seed=cfg.seed,
            cfg=cfg,
            results_dir=results_dir,
            ps=list(range(10, 100, 10)),
            save_images=True,
        )


    # Commit logs
    wandb.log(data={}, commit=True)




def train_autoencoder(
    cfg,
    fabric,
    rtpt,
    model: AbstractAutoencoder,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    warmup_scheduler,
    data_const,
    results_dir,
):
    """Trains the autoencoder model.

    Args:
        cfg: The configuration object.
        fabric: The Fabric object.
        rtpt: The RTPT object.
        model: The autoencoder model to train.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        optimizer: The optimizer to use for training.
        lr_scheduler: The learning rate scheduler to use.
        warmup_scheduler: The warmup scheduler to use.
        data_const: Constants derived from the data.
        results_dir: The directory to save results to.

    Returns:
        None
    """
    # Get evaluation iterations
    val_iterations = get_val_iterations(iterations_total=cfg.train.iters, num_val=cfg.num_val)
    image_generation_iterations = get_val_iterations(iterations_total=cfg.train.iters, num_val=cfg.num_val_images)

    # Add early stopping
    early_stopping = EarlyStopping(patience=cfg.num_val // 5)

    # Flag to check if data is image data
    is_image_data = not is_1d_data(cfg.dataset)

    pbar = tqdm.tqdm(train_loader, total=cfg.train.iters)
    for iteration, (data, target) in enumerate(pbar):
        if iteration == cfg.train.iters:
            break

        # Stop after a few batches in debug mode
        if cfg.debug and iteration > 2:
            break

        # Reset gradients
        optimizer.zero_grad()

        losses = model.loss(data)

        # Weight losses
        losses_weighted = {k: v * cfg.weight[f"{k}"] for k, v in losses.items()}

        # Sum up losses
        loss = sum(losses_weighted.values())

        # Compute gradients
        fabric.backward(loss)

        # Optional gradient clipping
        if cfg.train.clip_grad_norm > 0:
            # Clip gradients such that their total norm is no bigger than 2.0
            fabric.clip_gradients(model.decoder, optimizer, max_norm=cfg.train.clip_grad_norm)

        # Update weights
        optimizer.step()

        # If warmup is enabled, use warmup dampening context, else use nullcontext
        if cfg.train.warmup_enabled:
            warmup_context = warmup_scheduler.dampening
        else:
            warmup_context = nullcontext

        # Optional warmup context
        with warmup_context():
            # LR scheduler
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(metrics=losses["rec"])
            else:
                lr_scheduler.step()

        # Advance rtpt
        rtpt.step()


        # Logging
        if iteration % cfg.log_interval == 0:
            # Rewrite set_description with dict generator and string interpolation
            pbar.set_description(
                "Loss: "
                + " | ".join([f"{k}: {v.item():.4f}" for k, v in losses.items()])
                + (
                    f" | ES: [{early_stopping.counter}/{early_stopping.patience}] best: {early_stopping.best_val_loss:.2f}"
                    if cfg.train.early_stopping
                    else ""
                )
            )

            # Log learning rate
            wandb.log({"learning_rate/main": optimizer.param_groups[0]["lr"], "iteration": iteration}, commit=False)

            # Log wandb
            for k, v in losses.items():
                wandb.log({f"train_loss/{k}": v.item(), "iteration": iteration}, commit=False)

            # Commit logs
            if iteration not in val_iterations:
                # If this is an evaluation iteration, the logs are commited at a later point anyway
                wandb.log(data={}, commit=True)

        if (iteration in image_generation_iterations or cfg.debug) and is_image_data:
            make_images(
                model,
                data_const,
                iteration=iteration,
                seed=cfg.seed,
                cfg=cfg,
                results_dir=results_dir,
                ps=[50, 90],
            )

        if iteration in val_iterations or cfg.debug:
            # Compute validation loss
            val_losses = validation_loss(model, val_loader, "val", iteration=iteration, cfg=cfg)

            # Commit logs
            wandb.log(data={}, commit=True)

            # Check if early stopping should be triggered
            if cfg.train.early_stopping and early_stopping.check_stop(val_losses["val/rec"]):
                logger.info("Early stopping triggered. Stopping training now ...")
                break

    # Add validation loss one last time
    val_losses = validation_loss(model, val_loader, "val", iteration=cfg.train.iters, cfg=cfg)

    # Add images one last time
    if is_image_data:
        make_images(
            model,
            data_const,
            iteration=cfg.train.iters,
            seed=cfg.seed,
            cfg=cfg,
            results_dir=results_dir,
            ps=list(range(10, 100, 10)),
            save_images=True,
        )

    # Commit logs
    wandb.log(data={}, commit=True)
