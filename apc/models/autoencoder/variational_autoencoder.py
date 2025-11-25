import torch
from simple_einet.data import is_1d_data
from torch.nn import functional as F

from apc.models.autoencoder.vanilla_autoencoder import VanillaAutoencoder
from apc.models.encoder.nn_encoder import NeuralEncoder2D, NeuralEncoder2D, NeuralEncoder1D

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class VariationalAutoencoder(VanillaAutoencoder):

    def __reparameterize(self, mu, logvar, tau=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * tau

    def encode(self, x: torch.Tensor, mpe=True, tau=1.0) -> torch.Tensor:
        x = x.view(x.shape[0], *self.data_shape)

        mu, logvar = self.encoder(x).chunk(2, dim=1)

        if mpe:
            z = mu
        else:
            z = self.__reparameterize(mu=mu, logvar=logvar, tau=tau)

        return z

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:

        if self.training and self.cfg.nan_mask_strategy == "norm":
            # Update norm buffer with running mean
            self.norm.data = 0.99 * self.norm.data + 0.01 * x.mean(dim=0)


        if self.cfg.train.missing:
            min_p, max_p = self.cfg.train.missing_min,self.cfg.train.missing_max
            p = torch.rand(1).item() * (max_p - min_p) + min_p
            linspace = torch.linspace(0, 1, x.shape[2] * x.shape[3], device=x.device)
            rand = torch.rand(x.shape[0], linspace.shape[0], device=x.device)
            perm_indices = torch.argsort(rand, dim=1)
            rand = linspace[perm_indices].view(x.shape[0], 1, x.shape[2], x.shape[3])
            mask = rand < p

            # Repeat mask at dim=1 exactly x[1] times to ensure that the full pixel is selected
            mask = mask.repeat(1, x.shape[1], 1, 1)

            # x[mask] = self.cfg.nan_mask_value
            x_orig = x.clone()
            x = torch.where(mask, self.cfg.nan_mask_value * torch.ones_like(x), x)



        loss_dict = {}
        x = x.float()

        # DAE
        if self.cfg.dae_noise_std > 0 and self.training:
            x = self.add_noise(x)

        mu, logvar = self.encoder(x).chunk(2, dim=1)



        z = self.__reparameterize(mu, logvar)

        x_rec = self.decode(z)

        if self.cfg.train.missing:
            # Mask the reconstructed image as well, such that rec-error for missing pixels is zero
            x_rec = torch.where(mask, self.cfg.nan_mask_value * torch.ones_like(x_rec), x_rec)
            # Use original image for reconstruction loss to also reconstruct missing pixels
            # x = x_orig

        if self.cfg.train.rec_loss == "mse":
            loss_dict["rec"] = F.mse_loss(x_rec, x, reduction="sum") / x.shape[0]
        elif self.cfg.train.rec_loss == "bce":
            x = (x + 1) / 2 # Scale to [0, 1]
            x_rec = (x_rec + 1) / 2
            loss_dict["rec"] = F.binary_cross_entropy(x_rec, x, reduction="sum") / x.shape[0]
        else:
            raise ValueError(f"Unknown rec_loss: {self.cfg.train.rec_loss}")

        loss_dict["kld"] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

        return loss_dict
