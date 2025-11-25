#!/usr/bin/env python3

from contextlib import contextmanager, nullcontext
from omegaconf import DictConfig
from apc.losses import kl_divergence_independent_mv_gaussians
from simple_einet.abstract_layers import logits_to_log_weights
from simple_einet.layers.distributions.binomial import Binomial
from typing import Optional
from simple_einet.layers.distributions.normal import Normal
from simple_einet.sampling_utils import (
    index_one_hot,
    index_one_hot_fast,
    sample_categorical_differentiably,
    sampling_context,
)
from dataclasses import dataclass
from simple_einet.data import Shape, get_data_shape
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from functools import wraps
from torch.profiler import record_function


def prof(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with record_function(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@dataclass
class SamplingContext:
    """Dataclass for representing the context in which sampling operations occur."""

    # Number of samples
    num_samples: int = None

    # Indices into the out_channels dimension
    indices_out: torch.Tensor = None

    indices_repetition: torch.Tensor = None

    # MPE flag, if true, will perform most probable explanation sampling
    is_mpe: bool = False

    # Temperature for sampling at the leaves
    temperature_leaves: float = 1.0

    # Temperature for sampling at the einsumlayers
    temperature_sums: float = 1.0

    # Diff sampling method -- can be one of "simple" and "gumbel"
    diff_sampling_method: str = "simple"

    # Evidence
    evidence_data: torch.Tensor = None
    evidence_latents: torch.Tensor = None

    # Differentiable
    is_differentiable: bool = False

    # Temperature for differentiable sampling
    tau: float = 1.0

    # Do MPE at leaves
    mpe_at_leaves: bool = False

    # Return leaf distribution instead of samples
    return_leaf_params: bool = False

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"SamplingContext object has no attribute {key}")

    def __repr__(self) -> str:
        d = self.__dict__
        d2 = {}
        d2["evidence_data"] = d["evidence_data"].shape if d["evidence_data"] is not None else None
        d2["evidence_latents"] = d["evidence_latents"].shape if d["evidence_latents"] is not None else None
        d2["indices_out"] = d["indices_out"].shape if d["indices_out"] is not None else None
        return "SamplingContext(" + ", ".join([f"{k}={v}" for k, v in d2.items()]) + ")"


@contextmanager
def sampling_context(
    model: nn.Module,
    evidence_data: torch.Tensor = None,
    evidence_latents: torch.Tensor = None,
    requires_grad=False,
    seed=None,
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
            for module in model.modules():
                if hasattr(module, "_enable_input_cache"):
                    module._enable_input_cache()

            _ = model.log_likelihood(x=evidence_data, z=evidence_latents)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        if evidence_data is not None or evidence_latents is not None:
            for module in model.modules():
                if hasattr(module, "_enable_input_cache"):
                    module._disable_input_cache()

    if seed is not None:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)


class SumConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.logits = nn.Parameter(torch.log(torch.rand(out_channels, in_channels, kernel_size, kernel_size)))

        # Input cache
        self._is_input_cache_enabled = False
        self._input_cache: dict[str, Tensor] = {}

    def forward(self, x):

        if self._is_input_cache_enabled:
            self._input_cache["x"] = x

        N, C, H, W = x.size()
        K = self.kernel_size

        # x_og = x.clone()

        assert H % K == 0, f"Input height must be divisible by kernel size. Height was {H} and kernel size was {K}"
        assert W % K == 0, f"Input width must be divisible by kernel size. Width was {W} and kernel size was {K}"

        logits = self.logits
        log_weights = F.log_softmax(logits, dim=1)
        log_weights = log_weights.unsqueeze(0)

        # Make two new dimensions, such that the patches of KxK are now stacked
        x = x.view(N, C, H // K, K, W // K, K)
        x = x.permute(0, 1, 2, 4, 3, 5)  # 0 1 2 3 4 5 -> 0 1 2 4 3 5

        # Make space for out_channels
        x = x.view(N, 1, C, H // K, W // K, K, K)
        assert x.size() == (N, 1, C, H // K, W // K, K, K)

        # Make space in log_weights for H // kernel_size and W // kernel_size
        log_weights = log_weights.view(1, self.out_channels, self.in_channels, 1, 1, K, K)

        # Weighted sum over input channels
        x = torch.logsumexp(x + log_weights, dim=2)  # 0 1 2 4 3 5 -> 0 1 3 2 4

        # Invert permutation
        x = x.permute(0, 1, 2, 4, 3, 5)  # 0 1 2 4 3 5 -> 0 1 2 3 4 5
        x = x.contiguous().view(N, self.out_channels, H, W)
        return x

    def sample(self, ctx: SamplingContext) -> SamplingContext:
        # Select weights of this layer based on parent sampling path
        log_weights = self._select_weights(ctx, self.logits)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and len(self._input_cache) > 0:
            log_weights = self._condition_weights_on_evidence(ctx, log_weights)

        # Sample/mpe from the logweights
        indices = self._sample_from_weights(ctx, log_weights)

        ctx.indices_out = indices
        return ctx

    def _condition_weights_on_evidence(self, ctx, log_weights):
        input_cache_x = self._input_cache["x"]
        lls = input_cache_x
        log_prior = log_weights
        log_posterior = log_prior + lls
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)
        log_weights = log_posterior
        return log_weights

    def _select_weights(self, ctx, logits):
        N = ctx.num_samples
        Co, Ci, K, K = logits.shape
        H, W = ctx.indices_out.shape[2], ctx.indices_out.shape[3]

        # Index sums_out
        logits = logits.unsqueeze(0)  # make space for batch dim
        p_idxs = ctx.indices_out.unsqueeze(2)
        x = p_idxs

        x = x.view(N, Co, H // K, K, W // K, K)
        x = x.permute(0, 1, 2, 4, 3, 5)  # 0 1 2 3 4 5 -> 0 1 2 4 3 5

        # Make space for in_channels
        x = x.view(N, Co, 1, H // K, W // K, K, K)
        assert x.size() == (N, Co, 1, H // K, W // K, K, K)

        # Make space in log_weights for H // kernel_size and W // kernel_size
        logits = logits.view(1, Co, Ci, 1, 1, K, K)

        # Index into the "num_sums_out" dimension
        logits = index_one_hot_fast(logits, index=x, dim=1)

        assert logits.shape == (N, Ci, H // K, W // K, K, K)

        # Revert permutations etc
        logits = logits.permute(0, 1, 2, 4, 3, 5)  # 0 1 2 4 3 5 -> 0 1 2 3 4 5
        logits = logits.contiguous().view(N, Ci, H, W)

        log_weights = logits_to_log_weights(logits, dim=2, temperature=ctx.temperature_sums)
        return log_weights

    def _sample_from_weights(self, ctx, log_weights):
        indices = sample_categorical_differentiably(
            dim=1, is_mpe=ctx.is_mpe, hard=False, tau=ctx.tau, log_weights=log_weights, method=ctx.diff_sampling_method
        )
        return indices

    def _enable_input_cache(self):
        self._is_input_cache_enabled = True
        self._input_cache = {}

    def _disable_input_cache(self):
        self._is_input_cache_enabled = False
        self._input_cache.clear()


class SumLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, height: int, width: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.logits = nn.Parameter(torch.log(torch.rand(out_channels, in_channels, height, width) + 1e-10))
        self.num_features_out = height * width

        # Input cache
        self._is_input_cache_enabled = False
        self._input_cache: dict[str, Tensor] = {}

    def forward(self, x):

        if self._is_input_cache_enabled:
            self._input_cache["x"] = x

        logits = self.logits
        log_weights = F.log_softmax(logits, dim=1)
        log_weights = log_weights.unsqueeze(0)

        # Make space for out_channels in x
        x = x.unsqueeze(1)

        # Weighted sum over input channels
        x = torch.logsumexp(x + log_weights, dim=2)

        return x

    def _enable_input_cache(self):
        self._is_input_cache_enabled = True
        self._input_cache = {}

    def _disable_input_cache(self):
        self._is_input_cache_enabled = False
        self._input_cache.clear()

    def sample(self, ctx: SamplingContext) -> SamplingContext:
        # Select weights of this layer based on parent sampling path
        log_weights = self._select_weights(ctx, self.logits)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and len(self._input_cache) > 0:
            log_weights = self._condition_weights_on_evidence(ctx, log_weights)

        # Sample/mpe from the logweights
        indices = self._sample_from_weights(ctx, log_weights)

        ctx.indices_out = indices
        return ctx

    def _select_weights(self, ctx, logits):
        # Index sums_out
        logits = logits.unsqueeze(0)  # make space for batch dim
        p_idxs = ctx.indices_out.unsqueeze(2)
        x = p_idxs

        # Index into the "num_sums_out" dimension
        logits = index_one_hot_fast(logits, index=x, dim=1)

        log_weights = logits_to_log_weights(logits, dim=1, temperature=ctx.temperature_sums)

        return log_weights

    def _sample_from_weights(self, ctx, log_weights):
        indices = sample_categorical_differentiably(
            dim=1, is_mpe=ctx.is_mpe, hard=False, tau=ctx.tau, log_weights=log_weights, method=ctx.diff_sampling_method
        )
        return indices

    def _condition_weights_on_evidence(self, ctx, log_weights):
        input_cache_x = self._input_cache["x"]
        lls = input_cache_x
        log_prior = log_weights
        log_posterior = log_prior + lls
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True)
        log_weights = log_posterior
        return log_weights

    def __repr__(self):
        return f"SumLayer(ic={self.in_channels}, oc={self.out_channels}, h={self.height}, w={self.width})"



class ProdConv(nn.Module):
    def __init__(self, kernel_size_h: int, kernel_size_w: int, padding_w=0, padding_h=0):
        super().__init__()
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.padding_w = padding_w
        self.padding_h = padding_h

        # Input cache
        self._is_input_cache_enabled = False
        self._input_cache: dict[str, Tensor] = {}

    def forward(self, x):
        # Use a convolution with depthwise separable filters and ones as weights to simulate patch based product nodes
        # This is equivalent to a product node in the circuit
        N, C, H, W = x.size()
        Kh = self.kernel_size_h
        Kw = self.kernel_size_w

        assert (
            H + self.padding_h * 2
        ) % Kh == 0, f"Input height must be divisible by kernel size + 2*padding_h. Height was {H}, padding_h was {self.padding_h} and kernel size was {Kh}"
        assert (
            W + self.padding_w * 2
        ) % Kw == 0, f"Input width must be divisible by kernel size + 2*padding_w. Width was {W}, padding_w was {self.padding_w} and kernel size was {Kw}"

        # Construct a kernel with ones as weights
        ones = torch.ones(C, 1, Kh, Kw, device=x.device)

        # Apply the kernel to the input
        # x = F.conv2d(x, ones, groups=C, stride=(Kh, Kw), padding=0, bias=None)
        x = F.conv2d(x, ones, groups=C, stride=(Kh, Kw), padding=(self.padding_h, self.padding_w), bias=None)

        return x

    def _enable_input_cache(self):
        self._is_input_cache_enabled = True
        self._input_cache = {}

    def _disable_input_cache(self):
        self._is_input_cache_enabled = False
        self._input_cache.clear()

    def sample(self, ctx: SamplingContext):
        # Use repeat interleave to perform the following operation:
        #           1 1 2 2
        # 1 2  -\   1 1 2 2
        # 3 4  -/   3 3 4 4
        #           3 3 4 4
           
        idxs = ctx.indices_out

        idxs = torch.repeat_interleave(idxs, repeats=self.kernel_size_h, dim=-2)
        idxs = torch.repeat_interleave(idxs, repeats=self.kernel_size_w, dim=-1)

        # Cut off padding
        if self.padding_w != 0 and self.padding_h != 0:
            h, w = idxs.shape[-2:]
            idxs = idxs[..., self.padding_h : h - self.padding_h, self.padding_w : w - self.padding_w]


        ctx.indices_out = idxs

        return ctx

    def __repr__(self):
        return f"ProdConv(kh={self.kernel_size_h}, kw={self.kernel_size_w}, ph={self.padding_h}, pw={self.padding_w})"


class SumProdLayer(nn.Module):
    """Combines a SumLayer and a ProdConv in one layer. Implements forward and sample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size_h: int,
        kernel_size_w: int,
        order="sum-prod",
        sum_conv=False,
    ):
        super().__init__()
        if order == "sum-prod":
            if sum_conv:
                self.sum_layer = SumConv(in_channels, out_channels, kernel_size=kernel_size_w)
            else:
                self.sum_layer = SumLayer(in_channels, out_channels, height, width)
            self.prod_conv = ProdConv(kernel_size_h, kernel_size_w)
            self.num_features_out = height // kernel_size_h * width // kernel_size_w
            self.out_channels = out_channels
        elif order == "prod-sum":
            self.prod_conv = ProdConv(kernel_size_h, kernel_size_w)
            if sum_conv:
                self.sum_layer = SumConv(in_channels, out_channels, kernel_size=kernel_size_w)
            else:
                self.sum_layer = SumLayer(in_channels, out_channels, height // kernel_size_h, width // kernel_size_w)

        self.order = order

        self.num_features_out = height // kernel_size_h * width // kernel_size_w
        self.out_channels = out_channels

    def forward(self, x, return_intermediate=False):
        if self.order == "sum-prod":
            x_1 = self.sum_layer(x)
            x_2 = self.prod_conv(x_1)
        elif self.order == "prod-sum":
            x_1 = self.prod_conv(x)
            x_2 = self.sum_layer(x_1)
        else:
            raise ValueError(f"Order {self.order} not supported")
        if return_intermediate:
            return x_1, x_2
        else:
            return x_2

    def sample(self, ctx: SamplingContext):
        if self.order == "sum-prod":
            ctx = self.prod_conv.sample(ctx)
            ctx = self.sum_layer.sample(ctx)
        elif self.order == "prod-sum":
            ctx = self.sum_layer.sample(ctx)
            ctx = self.prod_conv.sample(ctx)
        return ctx

    def _enable_input_cache(self):
        self.sum_layer._enable_input_cache()

    def _disable_input_cache(self):
        self.sum_layer._disable_input_cache()

    def __repr__(self):
        if self.order == "sum-prod":
            return f"SumProdLayer({self.sum_layer}, {self.prod_conv})"
        elif self.order == "prod-sum":
            return f"ProdSumLayer({self.prod_conv}, {self.sum_layer})"
        else:
            raise ValueError(f"Order {self.order} not supported")


if __name__ == "__main__":

    pad = 1
    prod = ProdConv(2, 2, padding_w=pad, padding_h=pad)
    h = w = 6
    x = torch.ones(1, 1, h, w) * -0.5
    y = prod(x)
    print(x.shape)
    print(y.shape)
    print(x)
    print(y)
