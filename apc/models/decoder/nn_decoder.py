from omegaconf import DictConfig
from apc.enums import ModelName
import torch.nn.functional as F
import torch
from torch import nn

from apc.models.decoder.abstract_decoder import AbstractDecoder
from simple_einet.data import get_data_shape
from apc.models.components import ResidualStack, Up


import torch
from omegaconf import DictConfig
from torch import nn

from apc.models.decoder.abstract_decoder import AbstractDecoder
from simple_einet.data import get_data_shape

from apc.models.components import Up


class NeuralDecoder2D(AbstractDecoder):

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(cfg)

        h, w = self.data_shape.height, self.data_shape.width

        num_hidden = cfg.nn_decoder.num_hidden
        num_res_layers = cfg.nn_decoder.num_res_layers
        num_res_hidden = cfg.nn_decoder.num_res_hidden

        self.first_h = h // 2**self.cfg.nn_decoder.num_scales
        self.first_w = w // 2**self.cfg.nn_decoder.num_scales

        self.linear = nn.Linear(cfg.latent_dim, self.first_h * self.first_w * num_hidden)

        self._conv_1 = nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, stride=1, padding=1)

        self._residual_stack = ResidualStack(
            in_channels=num_hidden,
            num_hiddens=num_hidden,
            num_residual_layers=num_res_layers,
            num_residual_hiddens=num_res_hidden,
            bn=cfg.nn_decoder.bn,
        )
        assert self.cfg.nn_decoder.num_scales >= 2, "Number of scales must be greater than or equal to 2"

        layers = []
        for i in range(self.cfg.nn_decoder.num_scales - 2):
            layers.append(
                nn.ConvTranspose2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=4, stride=2, padding=1)
            )
        self.scales = nn.ModuleList(layers)

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hidden, out_channels=num_hidden // 2, kernel_size=4, stride=2, padding=1
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hidden // 2, out_channels=self.data_shape.channels, kernel_size=4, stride=2, padding=1
        )

    def decode(self, z: torch.Tensor, tau=1.0) -> torch.Tensor:
        z = z.view(z.size(0), self.cfg.latent_dim)

        # Linear layer
        x = self.linear(z)
        x = x.view(x.size(0), -1, self.first_h, self.first_w)

        x = self._conv_1(x)

        x = self._residual_stack(x)

        for scale in self.scales:
            x = scale(x)
            x = F.relu(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)

        # if self.cfg.model_name not in [ModelName.VAEM]:
        if self.cfg.model_name not in [ModelName.VAEM, ModelName.MIWAE]:
            x = torch.tanh(x)

        x = x.view(x.shape[0], *self.data_shape)

        return x


class NeuralDecoder1D(AbstractDecoder):

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(cfg=cfg)
        hidden_dim = cfg.nn_decoder.num_hidden
        self.latent_dim = cfg.latent_dim
        act_fn: object = lambda: nn.LeakyReLU(0.1)
        self.data_shape = get_data_shape(cfg.dataset)

        layers = [
            nn.Linear(self.latent_dim, hidden_dim * 2),
            act_fn(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            act_fn(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            act_fn(),
            nn.Linear(hidden_dim * 8, hidden_dim * 16),
            act_fn(),
            nn.Linear(hidden_dim * 16, self.data_shape.height),
        ]

        if self.cfg.nn_decoder.out_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.cfg.nn_decoder.out_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.cfg.nn_decoder.out_activation == "linear":
            pass  # No activation function
        else:
            raise ValueError(f"Unknown out_activation: {self.cfg.nn_decoder.out_activation}")

        self.net = nn.Sequential(*layers)

    def decode(self, z, tau=1.0):
        z = z.view(z.shape[0], self.latent_dim)
        x = self.net(z)
        return x.view(x.shape[0], *self.data_shape)


class DummyDecoder(AbstractDecoder):
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.randn(z.shape[0], *self.data_shape, device=z.device)


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config")
    def main_hydra(cfg: DictConfig):
        from torchsummary import summary

        ld = cfg.latent_dim
        encoder = NeuralDecoder2D(cfg)
        x = torch.randn(1, ld)
        print(encoder)
        print(encoder.decode(x).shape)

    main_hydra()
