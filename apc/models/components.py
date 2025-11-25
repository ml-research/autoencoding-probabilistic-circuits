#!/usr/bin/env python3


import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


class ResidualLayer(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


def Conv(
    mode: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    up: str = "conv",
) -> nn.Module:

    if mode == "down":
        layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
    elif mode == "up":

        if up == "conv":
            layer = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=1,
                bias=False,
            )
        elif up == "nearest":
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return nn.Sequential(
        layer,
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True),
    )


class Down(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, res: bool):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = Conv("down", in_channels, out_channels, stride=2)
        if res:
            self.res = ResidualLayer(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)

        if hasattr(self, "res"):
            x = self.res(x)
        return x


class Up(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, res: bool, up: str = "conv"):
        """
        Upsampling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            res (bool): Whether to use residual connection.
            up (str): Type of upsampling. Can be 'conv' or 'nearest'.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if res:
            self.res = ResidualLayer(in_channels)
        self.conv = Conv("up", in_channels, out_channels, stride=2, up=up)

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "res"):
            x = self.res(x)
        x = self.conv(x)
        return x



class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, bn=False):
        super(Residual, self).__init__()
        layers = [
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False),
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(num_residual_hiddens))
            layers.insert(5, nn.BatchNorm2d(num_hiddens))
        self._block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, bn):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens, bn) for _ in range(self._num_residual_layers)]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
