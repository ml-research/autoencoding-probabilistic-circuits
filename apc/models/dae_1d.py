#!/usr/bin/env python3
import warnings
from typing import List, Tuple

from omegaconf import DictConfig
from torch import nn
import numpy as np
import torch
from torch.nn import functional as F

from simple_einet.data import Shape, get_data_shape, is_1d_data


class ResBlock1D(nn.Module):
    def __init__(self, in_dim=64, out_dim=64):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        if in_dim != out_dim:
            self.expand = nn.Linear(in_dim, out_dim)
        else:
            self.expand = None

    def forward(self, x):
        identity = x
        if self.expand is not None:
            identity = self.expand(x)

        output = self.relu1(self.fc1(x))
        output = self.fc2(output)
        output = self.relu2(output + identity)
        return output


class Encoder1D(nn.Module):
    def __init__(self, in_shape: Shape, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.in_shape = in_shape
        self.hdim = latent_dim

        layers = []
        input_dim = in_shape.height
        for hdim in reversed(hidden_dims):
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(ResBlock1D(hdim, hdim))
            input_dim = hdim

        self.main = nn.Sequential(*layers)
        self.fc = nn.Linear(input_dim, latent_dim)
        self.out_shape = Shape(in_shape.channels, height=latent_dim, width=1)

    def forward(self, x):
        y = self.main(x)
        y = self.fc(y)
        return y


class VaeEncoder1D(nn.Module):
    def __init__(self, in_shape: Shape, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.in_shape = in_shape
        self.hdim = latent_dim

        layers = []
        input_dim = in_shape.height
        for hdim in reversed(hidden_dims):
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(ResBlock1D(hdim, hdim))
            input_dim = hdim

        self.main = nn.Sequential(*layers)
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)
        self.out_shape = Shape(in_shape.channels, height=latent_dim, width=1)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_shape[1])
        y = self.main(x)
        mu = self.mu(y)
        logvar = self.logvar(y)
        return mu, logvar


class Decoder1D(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], in_shape: Shape, out_shape: Shape, cfg: DictConfig):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.latent_dim = latent_dim
        self.cfg = cfg

        layers = []
        input_dim = latent_dim
        for hdim in hidden_dims[::-1]:
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.ReLU(True))
            layers.append(ResBlock1D(hdim, hdim))
            input_dim = hdim

        self.main = nn.Sequential(*layers)
        self.fc = nn.Linear(input_dim, out_shape.height)

    def forward(self, z):
        y = self.main(z)
        y = self.fc(y)
        if self.cfg.decoder.scale_output:
            y = torch.sigmoid(y)
            y = y * (2**self.cfg.n_bits - 1)
        return y.view(y.shape[0], *self.out_shape)


class AutoEncoder1D(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        data_shape: Shape,
        hidden_dims: List[int],
        upscale: bool = False,
        cfg: DictConfig = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.data_shape = data_shape

        if upscale:
            self.out_shape = Shape(data_shape.channels, height=data_shape.height * 2, width=1)
        else:
            self.out_shape = Shape(data_shape.channels, height=data_shape.height, width=1)

        hidden_dims_enc = hidden_dims[: int(np.log2(data_shape.height))]
        hidden_dims_dec = list(hidden_dims_enc)

        self.encoder = Encoder1D(
            in_shape=data_shape,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims_enc,
        )

        self.decoder = Decoder1D(
            in_shape=self.encoder.out_shape,
            latent_dim=latent_dim,
            out_shape=self.out_shape,
            hidden_dims=hidden_dims_dec,
            cfg=cfg,
        )

    def encode(self, x: torch.Tensor, mpe=True, mpe_at_leaves=True) -> torch.Tensor:
        if self.cfg.decoder.scale_output:
            x = x / (2**self.cfg.n_bits - 1)
        x = x.view(-1, self.data_shape.height)
        z = self.encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        z = self.encode(x)
        recon_x = self.decode(z)
        if self.cfg.decoder.scale_output:
            normalizer = 2**self.cfg.n_bits - 1
        else:
            normalizer = 1
        loss_rec = F.binary_cross_entropy(recon_x / normalizer, x / normalizer, reduction="sum") / x.shape[0]
        return {"rec": loss_rec}


class VariationalAutoencoder1D(nn.Module):
    def __init__(self, latent_dim: int, data_shape: Shape, hidden_dims: List[int], cfg: DictConfig):
        """Variational Autoencoder"""
        super(VariationalAutoencoder1D, self).__init__()
        self.latent_dim = latent_dim
        self.data_shape = data_shape
        self.cfg = cfg

        hidden_dims_enc = hidden_dims[: int(np.log2(data_shape.height))]
        hidden_dims_dec = list(hidden_dims_enc)

        self.encoder = VaeEncoder1D(
            in_shape=data_shape,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims_enc,
        )

        self.decoder = Decoder1D(
            in_shape=self.encoder.out_shape,
            latent_dim=latent_dim,
            out_shape=data_shape,
            hidden_dims=hidden_dims_dec,
            cfg=cfg,
        )

    def encode(self, x: torch.Tensor, mpe=True, mpe_at_leaves=True) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        if self.cfg.decoder.scale_output:
            x = x / (2**self.cfg.n_bits - 1)
        x = x.view(-1, self.data_shape.height)
        mu, logvar = self.encoder(x)

        if mpe or mpe_at_leaves:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)[0]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.float()
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        if self.cfg.decoder.scale_output:
            normalizer = 2**self.cfg.n_bits - 1
        else:
            normalizer = 1

        if is_1d_data(self.cfg.dataset):
            # 1D data is scaled into N(0, 1) -> use mse
            loss_rec = F.mse_loss(recon_x / normalizer, x / normalizer, reduction="sum") / x.shape[0]
        else:
            # Rest is in [0, 1] -> use BCE
            loss_rec = F.binary_cross_entropy(recon_x / normalizer, x / normalizer, reduction="sum") / x.shape[0]
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        return {"rec": loss_rec, "kld": loss_kld}


def setup_autoencoder_1d(
    cfg,
) -> Tuple[
    AutoEncoder1D,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    """
    This function sets up the AutoEncoder model, its optimizer and learning rate scheduler.

    Args:
        cfg: A configuration object containing the necessary parameters for the model, optimizer and scheduler.

    Returns:
        model: An instance of the AutoEncoder model.
        optimizer: An instance of the AdamW optimizer for the model parameters.
        lr_scheduler: A learning rate scheduler that adjusts the learning rate at certain milestones.

    Note: The learning rate scheduler uses a MultiStepLR strategy, where the learning rate is reduced by a factor of
    0.1 at 66% and 90% of the total training iterations.
    """
    # Get the shape of the data based on the dataset specified in the configuration
    data_shape = get_data_shape(cfg.dataset)

    # Initialize the AutoEncoder model with the configuration and data shape
    model = AutoEncoder1D(latent_dim=cfg.latent_dim, data_shape=data_shape, hidden_dims=cfg.decoder.channels, cfg=cfg)

    # Initialize the AdamW optimizer with the model parameters and learning rate specified in the configuration
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, amsgrad=cfg.train.amsgrad, weight_decay=cfg.train.weight_decay
    )

    # Initialize the learning rate scheduler with the optimizer and milestones specified in the configuration
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.66 * cfg.train.iters), int(0.9 * cfg.train.iters)], gamma=1e-1)

    # Return the model, optimizer and learning rate scheduler
    return model, optimizer, lr_scheduler


def setup_variational_autoencoder_1d(
    cfg,
) -> Tuple[
    VariationalAutoencoder1D,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    """
    This function sets up the VariationalAutoencoder model, its optimizer and learning rate scheduler.

    Args:
        cfg: A configuration object containing the necessary parameters for the model, optimizer and scheduler.

    Returns:
        model: An instance of the VariationalAutoencoder model.
        optimizer: An instance of the AdamW optimizer for the model parameters.
        lr_scheduler: A learning rate scheduler that adjusts the learning rate at certain milestones.

    Note: The learning rate scheduler uses a MultiStepLR strategy, where the learning rate is reduced by a factor of
    0.1 at 66% and 90% of the total training iterations.
    """
    # Get the shape of the data based on the dataset specified in the configuration
    data_shape = get_data_shape(cfg.dataset)

    # Initialize the AutoEncoder model with the configuration and data shape
    model = VariationalAutoencoder1D(
        latent_dim=cfg.latent_dim, data_shape=data_shape, hidden_dims=cfg.decoder.channels, cfg=cfg
    )

    # Initialize the AdamW optimizer with the model parameters and learning rate specified in the configuration
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, amsgrad=cfg.train.amsgrad, weight_decay=cfg.train.weight_decay
    )

    # Initialize the learning rate scheduler with the optimizer and milestones specified in the configuration
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.66 * cfg.train.iters), int(0.9 * cfg.train.iters)], gamma=1e-1)

    # Return the model, optimizer and learning rate scheduler
    return model, optimizer, lr_scheduler
