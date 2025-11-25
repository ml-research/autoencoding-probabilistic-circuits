"""
Adapted from: https://github.com/federicobergamin/MIWAE-PyTorch/blob/main/model.py
"""

import torch.nn.functional as F
from apc.enums import DecoderType
from apc.exp_utils import seed_context
from apc.models.encoder.nn_encoder import NeuralEncoder1D, NeuralEncoder2D
from apc.models.decoder.nn_decoder import NeuralDecoder1D, NeuralDecoder2D
from simple_einet.data import is_1d_data, is_debd_data, is_image_data
from apc.models.abstract_model import AbstractAutoencoder
import math

from omegaconf import DictConfig

import torch
import torch
import numpy as np
from torch import nn
from torch.distributions import Bernoulli, Binomial, Categorical, Independent, MixtureSameFamily, Normal
import matplotlib.pyplot as plt
from torchvision import utils
from math import ceil


class View(nn.Module):
    """For reshaping tensors inside Sequential objects"""

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


# class Encoder(nn.Module):
#     def __init__(self, channel_input: int, latent_dim: int):
#         super(Encoder, self).__init__()

#         self.channel_input = channel_input
#         self.latent_dim = latent_dim

#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#         )

#         self.output_layers = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2 * self.latent_dim))

#     def forward(self, x, n_samples=None):

#         _out = self.feature_extractor(x)
#         # print(_out.shape)
#         # I forget to flatten
#         _out = _out.view(-1, 4 * 4 * 32)
#         _out = self.output_layers(_out)

#         _mu = _out[:, 0 : self.latent_dim]
#         _log_var = _out[:, self.latent_dim :]

#         # reparametrization trick
#         dist = Normal(_mu, (0.5 * _log_var).exp())
#         if n_samples is None:
#             _z = dist.rsample()
#         else:
#             _z = dist.rsample([n_samples])

#         return _mu, _log_var, _z, dist


# class BernoulliDecoder(nn.Module):
#     def __init__(self, latent_dim: int, output_channel: int):
#         super(BernoulliDecoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.output_channel = output_channel

#         self.conv_layers = nn.Sequential(
#             nn.Linear(latent_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             View((-1, 32, 4, 4)),
#             nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
#         )

#     def forward(self, latent, n_samples=None):


#         # print(latent.shape)
#         _logits = self.conv_layers(latent)

#         # print('_logits shape after conv layer: ', _logits.shape) #[batch_size*n_sample, ]
#         if n_samples is not None:
#             _logits = _logits.view(
#                 n_samples, latent.shape[1], 28, 28
#             )  # here then logits are of the shape batch_size * iwae_samples * 28 * 28

#         output_dist = Bernoulli(logits=_logits)

#         return _logits, output_dist


class MIWAE(AbstractAutoencoder):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.input_channel = self.data_shape.channels
        self.latent_dim = self.cfg.latent_dim
        self.prior = Normal(0, 1)
        self.K_train = self.cfg.miwae.K_train
        self.K_test = self.cfg.miwae.K_test
        self.log_p_given_z = nn.BCEWithLogitsLoss()

    def _encoder(self, x: torch.Tensor, n_samples: int = None):

        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        # reparametrization trick
        dist = Normal(mu, (0.5 * logvar).exp())
        if n_samples is None:
            z = dist.rsample()
        else:
            z = dist.rsample([n_samples])
        return mu, logvar, z, dist

    def _decoder(self, z: torch.Tensor, n_samples: int = None):
        if n_samples is not None:
            bs = z.shape[1]
            z = z.view(bs * n_samples, self.latent_dim)

        logits = self.decoder(z)

        if n_samples is not None:
            logits = logits.view(n_samples, bs, *self.data_shape)
        # output_dist = Bernoulli(logits=logits)
        if is_debd_data(self.cfg.dataset):
            output_dist = Bernoulli(probs=logits)
        else:
            output_dist = Binomial(total_count=2**self.cfg.n_bits - 1, logits=logits)  # TODO: modify for tabular data
        return logits, output_dist

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

    def sample_z(self, num_samples: int, seed: int, tau: float = 1) -> torch.Tensor:
        with seed_context(seed):
            return tau * torch.randn(num_samples, self.cfg.latent_dim, device=self.mydevice)

    def fill_missing_(self, x: torch.Tensor) -> torch.Tensor:
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
        return x

    def encode(self, x: torch.Tensor, mpe: bool = True, tau: float = 1, n_samples: int = None) -> torch.Tensor:
        # Project x from (-1, 1) into (0, 1)
        x = (x + 1) / 2
        mu, _, z, _ = self._encoder(x, n_samples=n_samples)
        if mpe:
            return mu
        else:
            return z

    def decode(
        self, z: torch.Tensor, decoder_type: DecoderType = None, tau: float = 1, n_samples: int = None
    ) -> torch.Tensor:
        logits, _output_dist = self._decoder(z, n_samples=n_samples)

        # Turn into pixels
        # Back into (-1 , 1)
        pixels = torch.sigmoid(logits)
        if is_image_data(self.cfg.dataset):
            pixels = (pixels - 0.5) * 2
        else:
            pixels = logits
        return pixels

    def reconstruct(self, x: torch.Tensor, decoder_type: DecoderType = None, fill_evidence=False) -> torch.Tensor:
        M = 20
        batch_size = x.shape[0]
        original_data_shape = x.shape[1:]  # Store original shape

        # --- Step 1: Preparations (Masking, Filling, Encoding) ---
        mask = ~torch.isnan(x)  # True for observed values
        L = self.K_test  # Number of importance samples

        # Prepare input for encoder (fill NaNs)
        # Clone x to avoid modifying the original input tensor outside the function scope if fill_missing_ is in-place
        x_filled = x.clone()
        self.fill_missing_(x_filled)  # Fill NaNs with a placeholder value (e.g., 0 or mean)

        # Encode and get L latent samples z and variational distribution q(z|x)
        mu, log_var, z, q_dist = self._encoder(x_filled, n_samples=L)
        # z shape: [L, batch_size, latent_dim]
        # q_dist: Distribution object for q(z|x)

        # --- Step 2: Calculate Importance Weights ---
        # Calculate log p(z)
        logp = self.prior.log_prob(z)  # Shape: [L, batch_size, latent_dim]
        logp = torch.sum(logp, dim=-1)  # Shape: [L, batch_size]

        # Calculate log q(z|x)
        logq = q_dist.log_prob(z)  # Shape: [L, batch_size, latent_dim]
        logq = torch.sum(logq, dim=-1)  # Shape: [L, batch_size]

        # Decode z to get the generative distribution p(x|z)
        _, _output_dist = self._decoder(z, n_samples=L)
        # _output_dist: Distribution object for p(x|z) corresponding to L samples

        # Calculate log p(x_obs|z)
        # Prepare observed data x_obs for log_prob calculation (scaling/rounding)
        # Ensure this matches the assumptions of _output_dist.log_prob
        # Use the original x with NaNs for masking, but the scaled version for log_prob
        x_scaled = x_filled.clone()
        if is_image_data(self.cfg.dataset):
            x_scaled[mask] = (x_scaled[mask] + 1) / 2  # Scale observed values to [0, 1]
            x_scaled[mask] = torch.round(x_scaled[mask] * (2**self.cfg.n_bits - 1))  # Scale to [0, 2^n_bits - 1]
        # We only need log_prob for observed values, expand x_scaled for broadcasting
        x_scaled_expanded = x_scaled.unsqueeze(0).expand(L, -1, *original_data_shape)

        logpxz = _output_dist.log_prob(x_scaled_expanded)  # Shape: [L, batch_size, *data_shape]

        # Flatten data dimensions and apply mask
        logpxz_flat = logpxz.view(L, batch_size, -1)  # Shape: [L, batch_size, flattened_dims]
        mask_flat = mask.view(batch_size, -1)  # Shape: [batch_size, flattened_dims]

        # Zero-out log-probs for missing data points
        logpxobsgivenz = torch.sum(logpxz_flat * mask_flat.unsqueeze(0), dim=2)  # Shape: [L, batch_size]

        # Calculate normalized importance weights
        log_unnormalized_importance_weights = logpxobsgivenz + logp - logq
        # Ensure numerical stability for softmax
        log_unnormalized_importance_weights = (
            log_unnormalized_importance_weights - torch.max(log_unnormalized_importance_weights, dim=0, keepdim=True)[0]
        )
        imp_weights = torch.softmax(log_unnormalized_importance_weights, dim=0, dtype=torch.float32)
        # imp_weights shape: [L, batch_size]

        # --- Step 3: Generate L Candidate Samples ---
        # Sample L full datasets from the generative distribution p(x|z)
        x_samples = _output_dist.sample()  # Shape: [L, batch_size, *data_shape]

        # Post-process samples (e.g., scale back if necessary)
        # Assuming samples are in [0, 2^n_bits - 1] range if output_dist is appropriate
        if is_image_data(self.cfg.dataset):
            x_samples = x_samples / (2**self.cfg.n_bits - 1)  # Scale to [0, 1]
            x_samples = (x_samples - 0.5) * 2  # Scale to [-1, 1]
            x_samples = torch.clamp(x_samples, -1, 1)  # Clamp values

        # Flatten samples for resampling step
        x_samples_flat = x_samples.view(L, batch_size, -1)  # Shape: [L, batch_size, flattened_dims]

        # --- Step 4: Resampling Importance Sampling (SIR) ---
        # Prepare weights and samples for batch-wise resampling
        # Transpose weights: [batch_size, L]
        imp_weights_t = imp_weights.T
        # Transpose samples: [batch_size, L, flattened_dims]
        x_samples_flat_t = x_samples_flat.permute(1, 0, 2)

        # Sample M indices for each batch element with replacement, using imp_weights as probabilities
        # Add small epsilon for numerical stability if weights can be zero
        resampled_indices = torch.multinomial(imp_weights_t + 1e-10, M, replacement=True)
        # resampled_indices shape: [batch_size, M]

        # Gather the corresponding samples based on the resampled indices
        # Need to expand indices to match the dimensions of x_samples_flat_t for gather
        resampled_indices_expanded = resampled_indices.unsqueeze(-1).expand(-1, -1, x_samples_flat_t.shape[-1])
        # Gather along the L dimension (dim=1)
        imputations_flat = torch.gather(x_samples_flat_t, 1, resampled_indices_expanded)
        # imputations_flat shape: [batch_size, M, flattened_dims]

        # Permute to get [M, batch_size, flattened_dims]
        imputations_flat = imputations_flat.permute(1, 0, 2)

        # Reshape back to original data shape
        multiple_imputations = imputations_flat.reshape(M, batch_size, *original_data_shape)

        # --- Step 5: Optional - Fill Evidence ---
        if fill_evidence:
            # Use the original scaled x for filling evidence if needed
            x_original_scaled = x.clone()  # Use original x
            x_original_scaled = (x_original_scaled - 0.5) * 2  # Scale to [-1, 1]

            # Expand mask and original data for broadcasting
            mask_expanded = mask.unsqueeze(0).expand(M, -1, *original_data_shape)
            x_expanded = x_original_scaled.unsqueeze(0).expand(M, -1, *original_data_shape)

            # Fill observed values into each of the M imputations
            multiple_imputations[mask_expanded] = x_expanded[mask_expanded]

        return multiple_imputations.mean(0)

    def loss(self, x, mask_obs=None) -> dict[str, torch.Tensor]:

        # Create zeros mask with 50% randomly missing data
        # mask_obs = torch.rand(x.shape, device=x.device) < 1.0
        mask_obs = torch.rand(x.shape, device=x.device) < 0.5
        x[~mask_obs] = 0.0

        mu, log_var, z, q_dist = self._encoder(x, n_samples=self.K_train)

        # I have to compute the kl
        log_pz = self.prior.log_prob(z)

        log_qz = q_dist.log_prob(z)

        kl = log_qz - log_pz  # shape  [n_samples, batch_size, latent_dim]
        logits, output_dist = self._decoder(z, n_samples=self.K_train)  # shape batch_size * iwae_samples * 28 * 28

        # Project x from (-1, 1) into (0, 255)
        if is_image_data(self.cfg.dataset):
            x = (x + 1) / 2
            x = torch.round(x * (2**self.cfg.n_bits - 1))

        if self.K_train is not None:
            log_p_x_given_z = output_dist.log_prob(
                x.unsqueeze(0)
            )  # shape [batch_size, channel, 28, 28] or [batch_size * iwae_samples * 28 * 28]
        else:
            log_p_x_given_z = output_dist.log_prob(x)

        if self.K_train is not None:
            batch_kl = torch.mean(kl, axis=-1)  # .permute(1,0) # batch_size * n_samples

            if mask_obs is not None:
                # I have to be careful at multiplying the mask to the correct stuff
                masked_log_p_x_given_z = log_p_x_given_z * mask_obs.squeeze(0)
                batch_log_p_x_given_z = torch.sum(masked_log_p_x_given_z, axis=[2, 3, 4])  # batch_size * n_samples
            else:
                batch_log_p_x_given_z = torch.sum(log_p_x_given_z, axis=[2, 3, 4])  # batch_size * n_samples

            bound = batch_log_p_x_given_z - batch_kl  # batch_size * n_sample
            # now I have to take the log-sum-exp over the samples and the
            bound = torch.logsumexp(bound, axis=0) - math.log(self.K_train)  # batch_size shape
        else:
            batch_kl = torch.mean(kl, axis=1)

            if mask_obs is not None:
                masked_log_p_x_given_z = log_p_x_given_z * mask_obs
                batch_log_p_x_given_z = torch.sum(masked_log_p_x_given_z, axis=[1, 2, 3])
            else:
                batch_log_p_x_given_z = torch.sum(log_p_x_given_z, axis=[1, 2, 3])

            bound = batch_log_p_x_given_z - batch_kl  # I would like the bound to be [batch_size]

        avg_iwae_bound = torch.mean(bound)

        return {
            "rec": -1 * avg_iwae_bound,
        }

    def forward(self, x, n_sample=None, mask=None):
        z = self.encode(x)
        xhat = self.decode(z)

        return xhat

    def sample_from_prior(self, n_samples):

        _latent = self.prior.sample([n_samples, self.latent_dim])

        # now I have to pass those through the decoder
        _logits, output_dist = self._decoder(_latent, None)

        # now I have to transform these into probabilities
        probs = torch.sigmoid(_logits)
        samples = output_dist.sample()

        return probs, samples
