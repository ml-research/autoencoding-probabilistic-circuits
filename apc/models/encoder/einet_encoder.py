from typing import Tuple

import torch
from omegaconf import DictConfig

from apc.models.encoder.abstract_encoder import AbstractEncoder
from apc.pc_utils import get_leaf_args, get_latent_leaf_args
from simple_einet.data import get_data_shape
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.multidistribution import MultiDistributionLayer
from simple_einet.layers.distributions.normal import Normal


class EinetEncoder(AbstractEncoder):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.data_shape = get_data_shape(cfg.dataset)
        self.scopes_x = torch.arange(0, self.data_shape.num_pixels)
        self.scopes_z = torch.arange(self.data_shape.num_pixels, self.data_shape.num_pixels + self.latent_dim)
        self.pc = self._make_pc()

    def _make_pc(self) -> Einet:
        leaf_kwargs, leaf_type = get_leaf_args(cfg=self.cfg)
        leaf_kwargs_latent, leaf_type_latent = get_latent_leaf_args(cfg=self.cfg)

        leaf_kwargs = {
            "scopes_to_dist": [
                (self.scopes_x, leaf_type, leaf_kwargs),
                (self.scopes_z, leaf_type_latent, leaf_kwargs_latent),
            ]
        }
        config = EinetConfig(
            num_features=self.data_shape.num_pixels + self.cfg.latent_dim,
            num_channels=self.data_shape.channels,
            depth=self.cfg.einet.D,
            num_sums=self.cfg.einet.S,
            num_leaves=self.cfg.einet.I,
            num_repetitions=self.cfg.einet.R,
            num_classes=1,
            leaf_type=MultiDistributionLayer,
            leaf_kwargs=leaf_kwargs,
            layer_type=self.cfg.einet.layer,
            dropout=0.0,
            structure=self.cfg.einet.structure,
        )
        einet = Einet(config)
        return einet

    def encode(
        self, x: torch.Tensor, mpe: bool = True, return_leaf_params: bool = False, tau=1.0
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.view(x.shape[0], self.data_shape.channels, self.data_shape.num_pixels)

        # Sample z ~ p( z | x )
        # Construct evidence
        z_nan = self.scopes_z.new_zeros(x.shape[0], x.shape[1], self.cfg.latent_dim, device=x.device, dtype=torch.float)
        z_nan[:] = float("nan")

        evidence = torch.cat([x, z_nan], dim=2)

        x_and_z = self.pc.sample(
            evidence=evidence,
            is_differentiable=True,
            is_mpe=mpe,
            mpe_at_leaves=mpe,
            return_leaf_params=return_leaf_params,
            temperature_leaves=tau,
            temperature_sums=tau,
        )
        z = x_and_z[:, :, self.scopes_z]
        # Only take channel 0
        # z = z[:, [0], :]

        if return_leaf_params:
            mu, logvar = z[..., 0], z[..., 1]
            z = tau * torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
            return mu, logvar, z
        else:
            return z

    def decode(self, z: torch.Tensor, x: torch.Tensor | None = None, mpe: bool = True, tau=1.0, fill_evidence=False) -> torch.Tensor:
        # If x is given, we have partial evidence
        if x is not None:
            assert x.isnan().any(), "If x is given for decoding, it must contain NaNs."
            x_nan = x.view(-1, self.data_shape.channels, self.data_shape.num_pixels)
        else:
            # Construct empty evidence with nans
            x_nan = self.scopes_x.new_zeros(
                z.shape[0], self.data_shape.channels, self.data_shape.num_pixels, device=z.device, dtype=torch.float
            )
            x_nan[:] = float("nan")

        # Construct evidence
        evidence = torch.cat([x_nan, z,], dim=2,)

        # Sample from PC
        x_and_z = self.pc.sample(
            evidence=evidence,
            is_differentiable=True,
            is_mpe=mpe,
            mpe_at_leaves=mpe,
            temperature_leaves=tau,
            temperature_sums=tau,
        )
        x = x_and_z[:, :, self.scopes_x]
        return x.view(x.shape[0], *self.data_shape)

    def log_likelihood(self, x: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        if x is not None and z is not None:  # p(x, z)
            # Repeat second dim to data channels
            # z = z.repeat(1, self.data_shape.channels, 1)
            evidence = torch.cat(
                [
                    x.view(-1, self.data_shape.channels, self.data_shape.num_pixels),
                    z.view(-1, self.data_shape.channels, self.cfg.latent_dim),
                ],
                dim=2,
            )
            return self.pc(evidence)
        elif x is None and z is not None:  # p(z)
            evidence = torch.cat(
                [
                    self.scopes_x.new_zeros(
                        z.shape[0], self.data_shape.channels, self.data_shape.num_pixels, device=z.device
                    ),
                    z,
                ],
                dim=2,
            )
            return self.pc(evidence, marginalized_scopes=self.scopes_x)
        elif x is not None and z is None:  # p(x)
            evidence = torch.cat(
                [
                    x.view(x.shape[0], self.data_shape.channels, self.data_shape.num_pixels),
                    self.scopes_z.new_zeros(x.shape[0], self.data_shape.channels, self.cfg.latent_dim, device=x.device),
                ],
                dim=2,
            )
            return self.pc(evidence, marginalized_scopes=self.scopes_z)
        else:
            raise ValueError("Either x or z must be provided.")


    def sample(
            self,
            num_samples: int | None = None,
            evidence_data: torch.Tensor | None = None,
            evidence_latents: torch.Tensor | None = None,
            seed: int | None = None,
            mpe: bool = False,
            return_data: bool = True,
            return_latents: bool = False,
            return_leaf_params: bool = False,
            tau: float = 1.0,
            fill_evidence: bool = False,
            enable_input_cache=True,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x_and_z = self.pc.sample(
            num_samples=num_samples,
            evidence=evidence_data,
            seed=seed,
            is_mpe=mpe,
            mpe_at_leaves=mpe,
            fill_evidence=fill_evidence,
            return_leaf_params=return_leaf_params,
            temperature_leaves=tau,
            temperature_sums=tau,
        )
        x, z = x_and_z[:, :, self.scopes_x], x_and_z[:, :, self.scopes_z]

        match (return_data, return_latents):
            case (True, True):
                return x, z
            case (True, False):
                return x
            case (False, True):
                return z
            case (False, False):
                raise ValueError("At least one of return_data or return_latents must be True.")


    def sample_z(
        self,
        num_samples: int,
        seed: int,
        return_leaf_params: bool = False,
        is_differentiable: bool = True,
        tau: float = 1.0,
    ) -> torch.Tensor:
        z = self.pc.sample(
            num_samples=num_samples,
            seed=seed,
            marginalized_scopes=self.scopes_x,
            is_differentiable=is_differentiable,
            return_leaf_params=return_leaf_params,
            temperature_leaves=tau,
            temperature_sums=tau,
        )[:, :, self.scopes_z]
        return z
