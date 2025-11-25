from typing import Tuple

from contextlib import contextmanager, nullcontext
import numpy as np
from simple_einet.layers.distributions import AbstractLeaf
from simple_einet.abstract_layers import AbstractLayer
import torch
from omegaconf import DictConfig
from torch import nn

from apc.models.conv_pc import SumProdLayer, SumLayer, SamplingContext, ProdConv
from apc.models.encoder.abstract_encoder import AbstractEncoder
from simple_einet.layers.distributions.binomial import Binomial
from simple_einet.layers.distributions.normal import Normal, NormalMean, RatNormal
from simple_einet.layers.factorized_leaf import FactorizedLeafSimple
from simple_einet.layers.distributions.categorical import Categorical
from simple_einet.sampling_utils import index_one_hot, sample_categorical_differentiably


@contextmanager
def sampling_context(
    model: nn.Module,
    evidence_data: torch.Tensor = None,
    evidence_latents: torch.Tensor = None,
    requires_grad=False,
    seed=None,
    enable_input_cache=True
):
    """
    Context manager for sampling.

    If evidence is provdied, the SPN graph is reweighted with the likelihoods computed using the given evidence.

    Args:
        spn: SPN that is being used to perform the sampling.
        evidence: Provided evidence. The SPN will perform a forward pass prior to entering this contex.
        requires_grad: If False, runs in torch.no_grad() context. (default: False)
        marginalized_scopes: Scopes to marginalize. (default: None)
        seed: Seed to use for sampling. (default: None)

    """
    # If no gradients are required, run in no_grad context
    if not requires_grad:
        context = torch.no_grad
    else:
        # Else provide null context
        context = nullcontext

    if seed is not None:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        torch.manual_seed(seed)

    # Run forward pass in given context
    with context():
        if evidence_data is not None or evidence_latents is not None:
            # Enter
            if enable_input_cache:
                for module in model.modules():
                    if hasattr(module, "_enable_input_cache"):
                        module._enable_input_cache()

            _ = model.log_likelihood(x=evidence_data, z=evidence_latents)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        if evidence_data is not None or evidence_latents is not None:
            if enable_input_cache:
                for module in model.modules():
                    if hasattr(module, "_enable_input_cache"):
                        module._disable_input_cache()

    if seed is not None:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

class ProductLayer(nn.Module):

    def __init__(
        self,
        num_features: int,
    ):
        super().__init__()
        self.num_features = num_features
        assert self.num_features % 2 == 0, "num_features must be a multiple of 2"
        self.out_shape = f"(N, {self.num_features_out})"

    @property
    def num_features_out(self) -> int:
        return self.num_features // 2

    def forward(self, x: torch.Tensor):
        # Get left and right partition probs
        left = x[:, 0::2]
        right = x[:, 1::2]

        return left + right

    def sample(self, ctx: SamplingContext) -> SamplingContext:
        idx = ctx.indices_out
        indices = idx.view(ctx.num_samples, idx.shape[1], -1)
        indices = ctx.indices_out.repeat_interleave(2, dim=2)
        ctx.indices_out = indices.view(ctx.num_samples, idx.shape[1], -1)
        return ctx

    def extra_repr(self):
        return "num_features={}, num_features_out={}, out_shape={}".format(
            self.num_features,
            self.num_features_out,
            self.out_shape,
        )


import math

def compute_non_overlapping_kernel_and_padding(H_data, W_data, H_target, W_target):
    """
    Computes kernel size and padding such that a single F.conv2d with
    stride=kernel_size and dilation=1 transforms the input to the target size.

    Returns:
        kernel_size: (kH, kW)
        padding: (pH, pW)
    """
    if H_data <= 0 or W_data <= 0 or H_target <= 0 or W_target <= 0:
        raise ValueError("All dimensions must be positive.")

    # Compute required kernel sizes
    kH = math.ceil(H_data / H_target)
    kW = math.ceil(W_data / W_target)

    # Compute padding needed to make input + 2*padding divisible by kernel
    padded_H = kH * H_target
    padded_W = kW * W_target

    total_pad_H = max(padded_H - H_data, 0)
    total_pad_W = max(padded_W - W_data, 0)

    pH = total_pad_H // 2
    pW = total_pad_W // 2

    return (kH, kW), (pH, pW)


class ConvPcEncoder(AbstractEncoder):
    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)
        self.channels = cfg.conv_pc.channels
        self.depth = cfg.conv_pc.depth
        self.kernel_size_h = cfg.conv_pc.kernel_size_h
        self.kernel_size_w = cfg.conv_pc.kernel_size_w
        self.latent_dim = cfg.latent_dim

        # Data leaf
        kwargs = {
            "num_features": self.data_shape.num_pixels,
            "num_channels": self.data_shape.channels,
            "num_leaves": self.channels,
            "num_repetitions": 1,
        }
        if cfg.conv_pc.dist_data == "binomial":
            self.leaf_data = Binomial(
                total_count=2**cfg.n_bits - 1,
                **kwargs,
            )
        elif cfg.conv_pc.dist_data == "normal":
            self.leaf_data = Normal(
                **kwargs,
            )
        elif cfg.conv_pc.dist_data == "normal-mean":
            self.leaf_data = NormalMean(
                **kwargs,
            )
        elif cfg.conv_pc.dist_data == "normal_rat":
            self.leaf_data = RatNormal(
                min_sigma=cfg.normal_rat.min_std,
                max_sigma=cfg.normal_rat.max_std,
                min_mean=cfg.normal_rat.min_mean,
                max_mean=cfg.normal_rat.max_mean,
                **kwargs,
            )
        elif cfg.conv_pc.dist_data == "categorical":
            self.leaf_data = Categorical(
                num_bins=2**cfg.n_bits,
                **kwargs,
            )
        else:
            raise ValueError(f"Data distribution {cfg.conv_pc.dist_data} not supported")
        self.latent_depth = self.cfg.conv_pc.latent_depth
        self.latent_channels = self.cfg.conv_pc.latent_channels

        # Construct layers
        layers = []
        h, w = self.data_shape.height, self.data_shape.width
        kh, kw = self.kernel_size_h, self.kernel_size_w

        # Add sum layer on top
        layers.append(SumLayer(in_channels=self.channels, out_channels=1, height=1, width=1))

        # Build from top down
        h, w = 1, 1
        for i in reversed(range(0, self.depth - 1)):
            h, w = h * kh, w * kw
            layers.append(
                SumProdLayer(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    height=h,
                    width=w,
                    kernel_size_h=kh,
                    kernel_size_w=kw,
                    order=cfg.conv_pc.order,
                    sum_conv=cfg.conv_pc.sum_conv,
                )
            )

            if i == self.latent_depth:
                dim_at_latent_depth = h * w
                # Create permutation and inverse permutation for latent encoding
                if self.cfg.conv_pc.perm_latents:
                    self.register_buffer("latent_perm", torch.randperm(dim_at_latent_depth))
                    self.register_buffer("latent_perm_inv", torch.argsort(self.latent_perm))

                self.latent_sum_layer = SumLayer(
                    in_channels=self.latent_channels,
                    out_channels=self.channels,
                    height=h,
                    width=w,
                )

        # Add lowest ProdConv layer to reduce data height/width to h/w
        (kh, kw), (ph, pw) = compute_non_overlapping_kernel_and_padding(H_data=self.data_shape.height,
                                                                        W_data=self.data_shape.width,
                                                                        H_target=h,
                                                                        W_target=w,)

        layers.append(
            ProdConv(
                kernel_size_h=kh,
                kernel_size_w=kw,
                padding_h=ph,
                padding_w=pw,
            )
        )
        layers = reversed(layers)

        self.layers = nn.ModuleList(layers)

        # Latent leaf
        if cfg.conv_pc.dist_latent == "normal":
            self.leaf_latent = Normal(
                num_features=self.latent_dim,
                num_channels=1,
                num_leaves=self.latent_channels,
                num_repetitions=1,
            )
        elif cfg.conv_pc.dist_latent == "normal-mean":
            self.leaf_latent = NormalMean(
                num_features=self.latent_dim,
                num_channels=1,
                num_leaves=self.latent_channels,
                num_repetitions=1,
            )
        else:
            raise ValueError(f"Latent distribution {cfg.conv_pc.dist_latent} not supported")

        # If the latent dim is larger than the dim at the latent depth, we need to add product nodes
        latent_products_layers = []
        _dim = self.latent_dim
        while _dim > dim_at_latent_depth:
            # Add one more product layer which halvens the dim
            prod = ProductLayer(num_features=_dim)
            latent_products_layers.append(prod)
            _dim = prod.num_features_out
        self.latent_prod_layers = nn.ModuleList(latent_products_layers)



    @property
    def _has_latent_product_layers(self):
        return len(self.latent_prod_layers) > 0

    @property
    def __device(self):
        """Small hack to obtain the current device."""
        return next(self.parameters()).device

    def _forward_leaf_data(self, x):
        x = x.view(x.size(0), self.data_shape.channels, self.data_shape.height * self.data_shape.width)
        x = self.leaf_data(x, marginalized_scopes=None)
        x = x.view(x.size(0), self.data_shape.channels, self.data_shape.height, self.data_shape.width, self.channels)

        # Factorize data input channels
        x = x.sum(1)

        # Remove empty repetition dimension (artifact from Einet implementation)
        x = x.squeeze(-1)

        # Permute from (N, H, W, C) to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        return x

    def log_likelihood(self, x: torch.Tensor | None = None, z: torch.Tensor | None = None):

        if x is None:
            # Create nans
            x = torch.full(
                (z.size(0), self.data_shape.channels, self.data_shape.height, self.data_shape.width),
                fill_value=float("nan"),
                device=self.__device,
            )

        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        x = self._forward_leaf_data(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == self.latent_depth:

                if z is None:
                    # z is not given -> marginalize z == add 1.0 == add 0.0 in log-space == do nothing
                    continue

                # Obtain latent lls
                z = z.view(z.size(0), 1, self.latent_dim)
                z = self.leaf_latent(z, marginalized_scopes=None)
                z = z.view(z.size(0), self.latent_dim, self.latent_channels)

                if self._has_latent_product_layers:
                    for prod in self.latent_prod_layers:
                        z = prod(z)

                z = z.permute(0, 2, 1)

                # Create zeros vector of size h*w and randomly pack the latent variables into it
                latents = torch.zeros(x.size(0), self.latent_channels, x.size(2) * x.size(3), device=self.__device)
                latents[:, :, : self.latent_dim] = z
                if self.cfg.conv_pc.perm_latents:
                    latents = latents[:, :, self.latent_perm]
                latents = latents.view(x.size(0), self.latent_channels, x.size(2), x.size(3))
                latents = self.latent_sum_layer(latents)
                latents = latents.view_as(x)

                # Add the latent variables to the input (product node)
                x = x + latents

                pass

        return x.view(x.size(0))

    def _sample_latents(self, ctx: SamplingContext) -> torch.Tensor:
        # Save indices_out before sum layer to recover afterward
        indices_out_before_sum = ctx.indices_out
        ctx = self.latent_sum_layer.sample(ctx=ctx)

        if self._has_latent_product_layers:
            for prod in self.latent_prod_layers:
                ctx = prod.sample(ctx)

        # Sample latents
        indices_out = ctx.indices_out
        indices_out = indices_out.view(ctx.num_samples, indices_out.shape[1], -1)

        ctx.indices_out = None
        if ctx.return_leaf_params:
            z_samples = self.leaf_latent.get_params()
            z_samples = z_samples.squeeze(-2)  # Remove repetition dimension
        else:
            z_samples = self.leaf_latent.sample(ctx=ctx)
            z_samples = z_samples.unsqueeze(-1)

        # L is the number of parameters (e.g. normal dist has 2 parameters: mu and sigma, binomial has 1: p)
        _, C, HW, I, L = z_samples.shape
        N = ctx.num_samples

        ctx.indices_out = indices_out_before_sum

        # (N, C, HW, I, L) -> (N, I, C, HW, L)
        z_samples = z_samples.permute(0, 3, 1, 2, 4)

        latents = torch.zeros(
            N,
            self.latent_channels,
            1,
            indices_out.size(2),
            L,
            device=self.__device,
        )
        latents[..., : self.latent_dim, :] = z_samples
        latents = latents.view(*indices_out.shape, L)

        # Index into I
        indices_out = indices_out.unsqueeze(-1)  # Make space for L dimension
        z_samples = index_one_hot(latents, index=indices_out, dim=1)
        z_samples = z_samples.view(N, -1, L)  # flatten

        # # Invert permutation
        if self.cfg.conv_pc.perm_latents:
            z_samples = z_samples[..., self.latent_perm_inv, :]

        # Get only first self.latent_dim elements
        z_samples = z_samples[..., : self.latent_dim, :]

        return z_samples

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

        if num_samples is None and evidence_data is None and evidence_latents is None:
            raise ValueError("Either num_samples or evidence must be given.")
        if num_samples is None and evidence_data is not None:
            num_samples = evidence_data.size(0)
        if num_samples is None and evidence_latents is not None:
            num_samples = evidence_latents.size(0)

        if evidence_data is not None and evidence_latents is not None:
            assert evidence_data.size(0) == evidence_latents.size(
                0
            ), f"Batch size of evidence_data and evidence_latents must be equal. Got {evidence_data.size(0)} and {evidence_latents.size(0)}"

        ctx = SamplingContext(
            num_samples=num_samples,
            mpe_at_leaves=mpe,
            is_differentiable=True,
            evidence_data=evidence_data,
            evidence_latents=evidence_latents,
            return_leaf_params=return_leaf_params,
            is_mpe=mpe,
            temperature_leaves=tau,
            temperature_sums=tau,
            diff_sampling_method=self.cfg.apc.diff_sampling_method,
        )

        # Init indices_out: (N, C=1, H=1, W=1)
        ctx.indices_out = torch.ones(
            size=(num_samples, 1, 1, 1), dtype=torch.float, device=self.__device, requires_grad=True
        )

        ctx.indices_repetition = torch.ones(
            size=(num_samples, 1), dtype=torch.float, device=self.__device, requires_grad=True
        )

        with sampling_context(
            self,
            evidence_data=evidence_data,
            evidence_latents=evidence_latents,
            requires_grad=self.cfg.apc.diff_sampling,
            seed=seed,
            enable_input_cache=enable_input_cache
        ):

            # Iterate over layers
            for i, layer in reversed(list(enumerate(self.layers))):

                if i == self.latent_depth:

                    # Sample latents
                    z_samples = self._sample_latents(ctx)

                    if not return_data and return_latents:
                        return z_samples

                # Sample layer
                ctx = layer.sample(ctx=ctx)

            indices_out = ctx.indices_out
            ctx.indices_out = None
            samples = self.leaf_data.sample(ctx=ctx)

            # Samples are of shape (N, C, H*W, I)
            N, C, HW, I = samples.shape
            samples = samples.view(N, self.data_shape.channels, self.data_shape.height, self.data_shape.width, I)
            # (N, C, H, W, I) -> (N, I, C, H, W)
            samples = samples.permute(0, 4, 1, 2, 3)

            indices_out.unsqueeze_(2)  # Make space for in_channels (I) dim
            # Index into I
            samples = index_one_hot(samples, index=indices_out, dim=1)

        if evidence_data is not None and fill_evidence:
            # First make a copy such that the original object is not changed
            evidence_data = evidence_data.clone().float()
            shape_evidence_data = evidence_data.shape
            evidence_data = evidence_data.view_as(samples)
            mask = torch.isnan(evidence_data)
            evidence_data[mask] = samples[mask].to(evidence_data.dtype)
            evidence_data = evidence_data.view(shape_evidence_data)

            if return_latents:
                return evidence_data, z_samples
            else:
                return evidence_data
        else:
            if return_latents:
                return samples, z_samples
            else:
                return samples

    def encode(
        self, x: torch.Tensor, mpe=True, mpe_at_leaves=False, return_leaf_params=False, tau: float = 1.0
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.sample(
            evidence_data=x,
            mpe=mpe,
            return_leaf_params=return_leaf_params,
            return_latents=True,
            return_data=False,
            tau=tau,
            enable_input_cache=True,
        )

        if return_leaf_params:
            if self.cfg.conv_pc.dist_latent == "normal":
                mu, logvar = z[..., 0], z[..., 1]
                z = tau * torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
                return mu, logvar, z
            elif self.cfg.conv_pc.dist_latent == "normal-mean":
                mu = z[..., 0]
                z = tau * torch.randn_like(mu) + mu
                logvar = torch.zeros_like(mu)
                return mu, logvar, z
        else:
            return z

    def decode(
        self,
        z: torch.Tensor,
        x: torch.Tensor | None = None,
        mpe=True,
        mpe_at_leaves=True,
        tau: float = 1.0,
        return_leaf_params=False,
        fill_evidence=False,
    ) -> torch.Tensor:
        x = None  # We never want to use x when decoding (cheating)

        # Construct empty evidence with nans
        evidence_data = torch.zeros(
            z.size(0),
            self.data_shape.channels,
            self.data_shape.height,
            self.data_shape.width,
            device=z.device,
            dtype=z.dtype,
        )
        evidence_data[:] = float("nan")

        # Sample from PC
        x = self.sample(
            evidence_data=evidence_data,
            evidence_latents=z,
            mpe=mpe,
            return_data=True,
            return_leaf_params=return_leaf_params,
            tau=tau,
            fill_evidence=fill_evidence,
            enable_input_cache=True,
        )
        return x.view(-1, *self.data_shape)

    def sample_z(
        self, num_samples: int, seed: int, return_leaf_params=False, is_differentiable=True, tau: float = 1.0
    ) -> torch.Tensor:
        return self.sample(
            num_samples=num_samples,
            seed=seed,
            return_latents=True,
            return_data=False,
            return_leaf_params=return_leaf_params,
            tau=tau,
        )


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config")
    def main_hydra(cfg: DictConfig):
        encoder = ConvPcEncoder(cfg)
        x = torch.randint(low=0, high=255, size=(5, 1, 32, 32))
        z = torch.randn(5, cfg.latent_dim)
        z = encoder(x)
        x_rec = encoder.decode(z)
        print(x.shape)
        print(z.shape)
        print(x_rec.shape)

    main_hydra()
