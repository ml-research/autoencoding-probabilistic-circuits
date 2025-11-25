from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from torch import nn

from simple_einet.data import get_data_shape


class AbstractEncoder(nn.Module, ABC):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_shape = get_data_shape(cfg.dataset)
        self.latent_dim = cfg.latent_dim

    @abstractmethod
    def encode(self, x: torch.Tensor, mpe: bool = True, tau: float = 1.0) -> torch.Tensor:
        """Encode input data."""
        pass

    def forward(self, x: torch.Tensor, mpe: bool = True, tau: float = 1.0) -> torch.Tensor:
        """Forward pass."""
        return self.encode(x, mpe=mpe, tau=tau)
