import torch

from omegaconf import DictConfig

from apc.models.decoder.abstract_decoder import AbstractDecoder
from apc.models.encoder.abstract_encoder import AbstractEncoder
from apc.enums import DecoderType
from simple_einet.data import get_data_shape, is_1d_data
from torch import nn
from abc import ABC, abstractmethod


class AbstractAutoencoder(nn.Module, ABC):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_shape = get_data_shape(cfg.dataset)

        # Latent dim
        self.latent_dim = cfg.latent_dim

        # Flag to indicate if the input is 1d (tabular) or 2d (image) data
        self._is_1d_data = is_1d_data(self.cfg.dataset)

        # Create encoder and decoder
        self.encoder: AbstractEncoder = self.make_encoder()
        self.decoder: AbstractDecoder = self.make_decoder()

    @abstractmethod
    def make_encoder(self) -> AbstractEncoder:
        pass

    @abstractmethod
    def make_decoder(self) -> AbstractDecoder:
        pass

    @property
    def mydevice(self):
        """Small hack to obtain the current device."""
        return next(self.parameters()).device

    @abstractmethod
    def sample_z(
        self,
        num_samples: int,
        seed: int,
        tau: float = 1.0,
    ) -> torch.Tensor:
        pass

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add noise to input data. This is used for denoising autoencoders.

        Args:
            x: Input data.

        Returns:
            torch.Tensor: Noisy data.
        """
        noise = torch.randn_like(x) * self.cfg.dae_noise_std
        return x + noise

    def decoder_types(self) -> list[DecoderType]:
        """List of supported decoder types."""
        return [DecoderType.NN]

    def _default_decoder_type(self) -> DecoderType:
        """Default decoder type."""
        return self.decoder_types()[0]

    def sample_x(self, num_samples: int, seed: int, decoder_type: DecoderType = None, tau: float = 1.0) -> torch.Tensor:
        z = self.sample_z(num_samples, seed, tau=tau)
        x = self.decode(z, decoder_type=decoder_type, tau=tau)
        return x

    def encode(self, x: torch.Tensor, mpe: bool = True, tau: float = 1.0) -> torch.Tensor:
        """Encode input data."""
        x = x.view(x.shape[0], *self.data_shape)

        return self.encoder(x, mpe=mpe, tau=tau)

    def decode(self, z: torch.Tensor, decoder_type: DecoderType = None, tau: float = 1.0) -> torch.Tensor:
        """Decode input data."""
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor, decoder_type: DecoderType = None) -> torch.Tensor:
        """Reconstruct the input x."""
        z = self.encode(x, mpe=True)
        x_rec = self.decode(z, decoder_type=decoder_type)
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.reconstruct(x)


    @abstractmethod
    def loss(self, x) -> dict[str, torch.Tensor]:
        """
        Compute the loss of this model for a given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary with loss values.
        """
        pass

    def has_model_summary(self):
        return True

    def has_embeddings(self):
        return True


class AbstractPcAutoencoder(AbstractAutoencoder, ABC):

    @abstractmethod
    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        pass
