from typing import Tuple

import torch
from torch.nn import functional as F

from apc.models.abstract_model import AbstractAutoencoder
from apc.enums import DecoderType
from apc.models.encoder.nn_encoder import NeuralEncoder2D, NeuralEncoder2D, NeuralEncoder1D
from apc.models.decoder.nn_decoder import NeuralDecoder2D, NeuralDecoder2D, NeuralDecoder1D
from apc.exp_utils import seed_context
from simple_einet.data import is_1d_data

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class VanillaAutoencoder(AbstractAutoencoder):

    def __init__(self, cfg):
        super().__init__(cfg)

        if self.cfg.nan_mask_strategy == "norm":
            # Create layer norm for each feature
            self.register_buffer("norm", torch.zeros(size=[*self.data_shape]))

    def make_encoder(self) -> NeuralEncoder2D | NeuralEncoder1D:
        if is_1d_data(self.cfg.dataset):
            return NeuralEncoder1D(self.cfg)
        else:
            return NeuralEncoder2D(self.cfg)

    def make_decoder(self) -> NeuralDecoder2D | NeuralDecoder1D:
        if is_1d_data(self.cfg.dataset):
            return NeuralDecoder1D(self.cfg)
        else:
            return NeuralDecoder2D(self.cfg)

    def reconstruct(self, x: torch.Tensor, decoder_type: DecoderType = None, fill_evidence = True) -> torch.Tensor:
        # Get nan mask and replace with nan_mask_value
        mask = torch.isnan(x)

        if mask.any():
            if self.cfg.nan_mask_strategy == "value":
                x[mask] = self.cfg.nan_mask_value
            elif self.cfg.nan_mask_strategy == "mean":
                x[mask] = x[~mask].mean()
            elif self.cfg.nan_mask_strategy == "norm":
                x[mask] = self.norm.unsqueeze(0).expand(x.shape[0], *self.data_shape)[mask]
            else:
                raise ValueError(f"Unknown nan_mask_strategy: {self.cfg.nan_mask_strategy}")

        # Reconstruct
        x_rec = super().reconstruct(x, decoder_type=decoder_type)

        # Fill in evidence
        if mask.any() and fill_evidence:
            x_rec[~mask] = x[~mask]
        return x_rec

    def sample_z(self, num_samples: int, seed: int, tau: float = 1.0) -> torch.Tensor:
        with seed_context(seed):
            return tau * torch.randn(num_samples, self.cfg.latent_dim, device=self.mydevice)

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:

        if self.training and self.cfg.nan_mask_strategy == "norm":
            # Update norm buffer with running mean
            self.norm.data = 0.99 * self.norm.data + 0.01 * x.mean(dim=0)

        if self.cfg.dae_noise_std > 0 and self.training:
            # Denoising Autoencoder
            z = self.encode(self.add_noise(x))
        else:
            z = self.encode(x)

        x_rec = self.decode(z)

        if self.cfg.train.rec_loss == "mse":
            loss_rec = F.mse_loss(x_rec, x, reduction="sum") / x.shape[0]
        elif self.cfg.train.rec_loss == "bce":
            loss_rec = F.binary_cross_entropy(x_rec, x, reduction="sum") / x.shape[0]
        else:
            raise ValueError(f"Unknown rec_loss: {self.cfg.train.rec_loss}")


        return {"rec": loss_rec}
