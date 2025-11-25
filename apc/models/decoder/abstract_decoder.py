from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from torch import nn

from simple_einet.data import get_data_shape


class AbstractDecoder(nn.Module, ABC):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_shape = get_data_shape(cfg.dataset)

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode input data."""
        pass

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.decode(z)
