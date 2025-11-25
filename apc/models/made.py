"""
Source: https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/made.py#L122
"""

# /usr/bin/env python3
from torch import nn
import torch

import numpy as np


class MaskedLinear(nn.Linear):
    """A Linear layer with masks that turn off some of the layer's weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones((out_features, in_features)))

    def set_mask(self, mask):
        self.mask.data.copy_(mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class MADE(nn.Module):
    """The Masked Autoencoder Distribution Estimator (MADE) model."""

    def __init__(self, input_dim, hidden_dims=None, n_masks=1):
        """Initializes a new MADE instance.

        Args:
            input_dim: The dimensionality of the input.
            hidden_dims: A list containing the number of units for each hidden layer.
            n_masks: The total number of distinct masks to use during training/eval.
        """
        super().__init__()
        self._input_dim = input_dim
        self._dims = [self._input_dim] + (hidden_dims or []) + [self._input_dim * 2]
        self._n_masks = n_masks
        self._mask_seed = 0

        layers = []
        for i in range(len(self._dims) - 1):
            in_dim, out_dim = self._dims[i], self._dims[i + 1]
            layers.append(MaskedLinear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self._net = nn.Sequential(*layers[:-1])

    def _sample_fn(self, mu, logvar):
        return torch.normal(mu, 0.5 * torch.exp(logvar))

    def _sample_masks(self):
        """Samples a new set of autoregressive masks.

        Only 'self._n_masks' distinct sets of masks are sampled after which the mask
        sets are rotated through in the order in which they were sampled. In
        principle, it's possible to generate the masks once and cache them. However,
        this can lead to memory issues for large 'self._n_masks' or models many
        parameters. Finally, sampling the masks is not that computationally
        expensive.

        Returns:
            A tuple of (masks, ordering). Ordering refers to the ordering of the outputs
            since MADE is order agnostic.
        """
        rng = np.random.RandomState(seed=self._mask_seed % self._n_masks)
        self._mask_seed += 1

        # Sample connectivity patterns.
        conn = [rng.permutation(self._input_dim)]
        for i, dim in enumerate(self._dims[1:-1]):
            # NOTE(eugenhotaj): The dimensions in the paper are 1-indexed whereas
            # arrays in Python are 0-indexed. Implementation adjusted accordingly.
            low = 0 if i == 0 else np.min(conn[i - 1])
            high = self._input_dim - 1
            conn.append(rng.randint(low, high, size=dim))
        conn.append(np.copy(conn[0]))

        # Create masks.
        masks = [conn[i - 1][None, :] <= conn[i][:, None] for i in range(1, len(conn) - 1)]
        masks.append(conn[-2][None, :] < conn[-1][:, None])

        # handle the case where nout = nin * k, for integer k > 1
        if (k := self._dims[-1] // self._dims[0]) > 1:
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=0)

        return [torch.from_numpy(mask.astype(np.uint8)) for mask in masks], conn[-1]

    def _forward(self, x, masks):
        layers = [layer for layer in self._net.modules() if isinstance(layer, MaskedLinear)]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)
        return self._net(x)

    def forward(self, x):
        """Computes the forward pass.

        Args:
            x: Either a tensor of vectors with shape (n, input_dim) or images with shape
                (n, 1, h, w) where h * w = input_dim.
        Returns:
            The result of the forward pass.
        """

        masks, _ = self._sample_masks()
        return self._forward(x, masks)

    @torch.no_grad()
    def sample(self, n_samples, conditioned_on=None):
        """See the base class."""
        conditioned_on = self._get_conditioned_on(n_samples, conditioned_on)
        return self._sample(conditioned_on)

    def _get_conditioned_on(self, n_samples, conditioned_on):
        assert (
            n_samples is not None or conditioned_on is not None
        ), 'Must provided one, and only one, of "n_samples" or "conditioned_on"'
        if conditioned_on is None:
            conditioned_on = torch.ones(n_samples, self._input_dim) * -1
        else:
            conditioned_on = conditioned_on.clone()
        return conditioned_on

    def _sample(self, x):
        masks, ordering = self._sample_masks()
        ordering = np.argsort(ordering)
        for dim in ordering:
            out = self._forward(x, masks)
            mu = out[:, dim]
            logvar = out[:, self._input_dim + dim]
            out = self._sample_fn(mu, logvar)
            x[:, dim] = torch.where(x[:, dim] < 0, out, x[:, dim])
        return x


