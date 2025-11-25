import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
import warnings

from apc.models.encoder.abstract_encoder import AbstractEncoder
from apc.enums import ModelName
from simple_einet.data import get_data_shape

from apc.models.components import Down, ResidualStack




class NeuralEncoder2D(AbstractEncoder):

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(cfg)

        h, w = self.data_shape.height, self.data_shape.width
        in_channels = self.data_shape.channels

        num_hidden = cfg.nn_encoder.num_hidden
        num_residual_layers = cfg.nn_encoder.num_res_layers
        num_residual_hidden = cfg.nn_encoder.num_res_hidden

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=num_hidden // 2, kernel_size=4, stride=2, padding=1
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hidden // 2, out_channels=num_hidden, kernel_size=4, stride=2, padding=1
        )
        assert self.cfg.nn_encoder.num_scales >= 2, f"Number of scales should be equal or greater than 2 but was {self.cfg.nn_encoder.num_scales}"
        layers = []
        for i in range(self.cfg.nn_encoder.num_scales - 2):
            layers.append(
                nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=4, stride=2, padding=1)
            )
        self.scales = nn.ModuleList(layers)

        self._conv_3 = nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, stride=1, padding=1)

        self._residual_stack = ResidualStack(
            in_channels=num_hidden,
            num_hiddens=num_hidden,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hidden,
            bn=cfg.nn_encoder.bn,
        )


        linear_in = h // 2 ** self.cfg.nn_encoder.num_scales * w // 2 ** self.cfg.nn_encoder.num_scales * num_hidden
        linear_out = self.latent_dim

        if self.cfg.model_name in [ModelName.VAE, ModelName.VAEM, ModelName.MIWAE, ModelName.HIVAE]:
            self._conv_4 = nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden // 2, kernel_size=3, stride=1, padding=1)
            linear_in = linear_in // 2
            linear_out = linear_out * 2

        if linear_in <= linear_out:
            warnings.warn(
                f"Encoder linear layer should reduce the dimensionality but got Linear(in_features={linear_in}, out_features={linear_out})."
            )

        self.linear = nn.Linear(linear_in, linear_out)

    def encode(self, x: torch.Tensor, mpe: bool = True, tau=1.0) -> torch.Tensor:
        x = x.view(x.shape[0], self.data_shape.channels, self.data_shape.height, self.data_shape.width)
        # Impute NaNs with 0
        mask = torch.isnan(x)
        x[mask] = self.cfg.nan_mask_value

        x = self._conv_1(x)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        for scale in self.scales:
            x = scale(x)
            x = F.relu(x)

        x = self._conv_3(x)
        x = self._residual_stack(x)

        if self.cfg.model_name in [ModelName.VAE, ModelName.VAEM, ModelName.MIWAE,ModelName.HIVAE, ModelName.HIVAE]:
            x = self._conv_4(x)

        # Flatten
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class NeuralEncoder1D(AbstractEncoder):

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(cfg=cfg)
        hidden_dim = cfg.nn_encoder.num_hidden
        act_fn = lambda: nn.LeakyReLU(0.1)
        self.data_shape = get_data_shape(cfg.dataset)

        self.last_layer_dim = self.cfg.latent_dim if self.cfg.model_name == ModelName.AE else self.cfg.latent_dim * 2

        self.net = nn.Sequential(
            nn.Linear(self.data_shape.height, hidden_dim * 16),
            act_fn(),
            nn.Linear(hidden_dim * 16, hidden_dim * 8),
            act_fn(),
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            act_fn(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            act_fn(),
            nn.Linear(hidden_dim * 2, self.last_layer_dim),
        )

    def encode(self, x, mpe: bool = True, tau=1.0):
        mask = torch.isnan(x)
        x[mask] = self.cfg.nan_mask_value
        x = x.view(x.shape[0], self.data_shape.num_pixels)  # Flatten the input
        x = self.net(x)
        x = x.view(x.shape[0], self.last_layer_dim)
        return x


class DummyEncoder(AbstractEncoder):
    def encode(self, x: torch.Tensor, mpe: bool = True, tau: float = 1.0) -> torch.Tensor:
        return torch.randn(x.shape[0], self.latent_dim, device=x.device)


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config")
    def main_hydra(cfg: DictConfig):
        from torchsummary import summary

        encoder = NeuralEncoder2D(cfg)
        print(encoder)
        x = torch.randn(1, 3, 32, 32)
        y = encoder(x)
        print(y.shape)

        summary(encoder, (1, 32, 32))

    main_hydra()
