from typing import Tuple, List

from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from apc.models.encoder.abstract_encoder import AbstractEncoder
from apc.pc_utils import get_leaf_args
from apc.enums import PcEncoderType, DecoderType
from simple_einet.data import get_data_shape
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.linsum import LinsumLayer
from simple_einet.layers.mixing import MixingLayer
from simple_einet.layers.sum import SumLayer
from simple_einet.sampling_utils import SamplingContext


@contextmanager
def sampling_context_act_einet(
    model: nn.Module,
    evidence_data: torch.Tensor = None,
    requires_grad=False,
    seed=None,
):
    """
    NOTE: This is a modified version of the sampling_context that only activates input caches for the activation extraction layers.

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
        if evidence_data is not None:
            # Instead of enabling input cache for all layers, we only enable it for the
            # layers that were selected for the activation extraction
            model.layers[model.layer_extraction_idx]._enable_input_cache()

            _ = model(x=evidence_data)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        if evidence_data is not None:
            model.layers[model.layer_extraction_idx]._disable_input_cache()

    if seed is not None:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

def _pc_sample(
    pc,
    num_samples: int = None,
    class_index=None,
    z: torch.Tensor = None,
    is_mpe: bool = False,
    mpe_at_leaves: bool = False,
    temperature_leaves: float = 1.0,
    temperature_sums: float = 1.0,
    marginalized_scopes: list[int] = None,
    is_differentiable: bool = False,
    return_leaf_params: bool = False,
    seed: int = None,
    fill_evidence: bool = False,
    cfg: DictConfig = None,
):
    """
    NOTE: Modified version of Einet.sample to account for sampling_context_act context manager.

    Sample from the distribution represented by this SPN.

    Possible valid inputs:

    - `num_samples`: Generates `num_samples` samples.
    - `num_samples` and `class_index (int)`: Generates `num_samples` samples from P(X | C = class_index).
    - `class_index (List[int])`: Generates `len(class_index)` samples. Each index `c_i` in `class_index` is mapped
        to a sample from P(X | C = c_i)
    - `evidence`: If evidence is given, samples conditionally and fill NaN values.

    Args:
        num_samples: Number of samples to generate.
        class_index: Class index. Can be either an int in combination with a value for `num_samples` which will result in `num_samples`
            samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
            list to a sample from P(X | C = c_i).
        z: Evidence that can be provided to condition the samples. If evidence is given, `num_samples` and
            `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
            distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
            sampled values.
        is_mpe: Flag to perform max sampling (MPE).
        mpe_at_leaves: Flag to perform mpe only at leaves.
        marginalized_scopes: List of scopes to marginalize.
        is_differentiable: Flag to enable differentiable sampling.
        return_leaf_params: Flag to return the leaf distribution instead of the samples.
        seed: Seed for torch.random.

    Returns:
        torch.Tensor: Samples generated according to the distribution specified by the SPN.

    """
    assert class_index is None or z is None, "Cannot provide both, evidence and class indices."
    assert (
            num_samples is None or z is None
    ), "Cannot provide both, number of samples to generate (num_samples) and evidence."
    if pc.config.num_classes == 1:
        assert class_index is None, "Cannot sample classes for single-class models (i.e. num_classes must be 1)."

    # Check if evidence contains nans
    if z is not None:
        # Set n to the number of samples in the evidence
        num_samples = z.shape[0]
    elif num_samples is None:
        num_samples = 1

    if is_differentiable:
        indices_out = torch.ones(size=(num_samples, 1, 1), dtype=torch.float, device=pc._device, requires_grad=True)
        indices_repetition = torch.ones(size=(num_samples, 1), dtype=torch.float, device=pc._device, requires_grad=True)
    else:
        indices_out = torch.zeros(size=(num_samples, 1), dtype=torch.long, device=pc._device)
        indices_repetition = torch.zeros(size=(num_samples,), dtype=torch.long, device=pc._device)


    has_latents = z is not None

    if has_latents and cfg.apc.encoder == PcEncoderType.EINET_ACT:
        if pc.layer_extraction_idx  == (len(pc.layers) - 1) and pc.config.num_repetitions > 1:
            # Activations were input to mixing layer --> set input cache for mixing layer
            activated_layer = pc.mixing
        else:
            # Set evidence as input cache for the sum layer above
            if pc.layer_extraction_idx == len(pc.layers) - 2:
                activated_layer = pc.layers[pc.layer_extraction_idx + 1]
            else:
                activated_layer = pc.layers[pc.layer_extraction_idx]

        activated_layer._enable_input_cache()
        activated_layer._input_cache["x"] = z

    ctx = SamplingContext(
        num_samples=num_samples,
        is_mpe=is_mpe,
        mpe_at_leaves=mpe_at_leaves,
        temperature_leaves=temperature_leaves,
        temperature_sums=temperature_sums,
        num_repetitions=pc.config.num_repetitions,
        evidence=z,
        indices_out=indices_out,
        indices_repetition=indices_repetition,
        is_differentiable=is_differentiable,
        return_leaf_params=return_leaf_params,
    )
    # Save parent indices that were sampled from the sampling root
    if pc.config.num_repetitions > 1:
        indices_out_pre_root = ctx.indices_out
        ctx = pc.mixing.sample(ctx=ctx)

        # Obtain repetition indices
        if is_differentiable:
            ctx.indices_repetition = ctx.indices_out.view(num_samples, pc.config.num_repetitions)
        else:
            ctx.indices_repetition = ctx.indices_out.view(num_samples)
        ctx.indices_out = indices_out_pre_root

    # Iterate over layers
    for i, layer in reversed(list(enumerate(pc.layers))):
        match cfg.apc.encoder:
            case PcEncoderType.EINET_ACT:
                # ACT encoding: simply sample, we already inserted the activation (z) above
                ctx = layer.sample(ctx=ctx)
            case PcEncoderType.EINET_CAT:
                # CAT encoding
                if has_latents and i == pc.layer_extraction_idx:
                    if i == len(pc.layers) - 1 and pc.config.num_repetitions > 1:
                        # Encoding is from the layer below the repetition mixing layer
                        ctx.indices_repetition = z
                    else:
                        # Insert CAT encoding
                        ctx.indices_out = z

                # Sample layer
                ctx = layer.sample(ctx=ctx)
            case _:
                raise ValueError(f"Unknown encoder {cfg.apc.encoder}.")


    if has_latents and cfg.apc.encoder == PcEncoderType.EINET_ACT:
        activated_layer._disable_input_cache()

    # Apply inverse permutation
    if hasattr(pc, "permutation_inv"):
        ctx.indices_out = ctx.indices_out[:, pc.permutation_inv]

    # Sample leaf
    samples = pc.leaf.sample(ctx=ctx)

    if return_leaf_params:
        # Samples contain the distribution parameters instead of the samples
        return samples

    if z is not None and fill_evidence:
        # First make a copy such that the original object is not changed
        z = z.clone().float()
        shape_evidence = z.shape
        z = z.view_as(samples)
        if marginalized_scopes is None:
            mask = torch.isnan(z)
            z[mask] = samples[mask].to(z.dtype)
        else:
            z[:, :, marginalized_scopes] = samples[:, :, marginalized_scopes].to(z.dtype)

        z = z.view(shape_evidence)
        return z
    else:
        return samples

class EinetSpnae(Einet):
    def __init__(self, config: EinetConfig, latent_dim: int = None):
        """
        Create an Einet based on a configuration object.

        This class contains modifications for SPNAE

        Args:
            config (EinetConfig): Einet configuration object.
            latent_dim (int): Dimensionality of the latent space.
        """
        if latent_dim is not None:
            assert latent_dim > 0, "Latent dimension must be positive."

        self.latent_dim = latent_dim
        super().__init__(config)




    def _build_structure_original(self):

        intermediate_layers: List[LinsumLayer] = []

        # Construct layers from top to bottom
        for i in np.arange(start=1, stop=self.config.depth + 1):
            # Choose number of input sum nodes
            # - if this is an intermediate layer, use the number of sum nodes from the previous layer
            # - if this is the first layer, use the number of leaves as the leaf layer is below the first sum layer
            if i < self.config.depth:
                _num_sums_in = self.config.num_sums
            else:
                _num_sums_in = self.config.num_leaves

            # Choose number of output sum nodes
            # - if this is the last layer, use the number of classes
            # - otherwise use the number of sum nodes from the next layer
            if i == 1:
                if self.latent_dim is not None:
                    _num_sums_out = self.latent_dim
                else:
                    _num_sums_out = self.config.num_classes
            else:
                _num_sums_out = self.config.num_sums

            # Calculate number of input features: since we represent a binary tree, each layer merges two partitions,
            # hence viewing this from top down we have 2**i input features at the i-th layer
            in_features = 2**i

            layer = LinsumLayer(
                num_features=in_features,
                num_sums_in=_num_sums_in,
                num_sums_out=_num_sums_out,
                num_repetitions=self.config.num_repetitions,
                dropout=self.config.dropout,
            )

            intermediate_layers.append(layer)

        if self.latent_dim is not None:
            # Add sum layer from latent space to 1
            layer = SumLayer(
                num_sums_in=self.latent_dim,
                num_features=1,
                num_sums_out=1,
                num_repetitions=self.config.num_repetitions,
                dropout=self.config.dropout,
            )
            intermediate_layers.insert(0, layer)

        # Construct leaf
        leaf_num_features_out = intermediate_layers[-1].num_features
        self.leaf = self._build_input_distribution(num_features_out=leaf_num_features_out)

        # List layers in a bottom-to-top fashion
        self.layers: List[LinsumLayer] = nn.ModuleList(reversed(intermediate_layers))

        # If model has multiple reptitions, add repetition mixing layer
        if self.config.num_repetitions > 1:
            self.mixing = MixingLayer(
                num_features=1,
                num_sums_in=self.config.num_repetitions,
                num_sums_out=self.config.num_classes,
                dropout=self.config.dropout,
            )


class EinetSpnaeEncoder(AbstractEncoder):
    """
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.pc = self._make_pc()

        # We find the layer activation extraction index by going through all layers and finding the layer with the num_features_out that matches the latent_dim
        self.layer_extraction_idx = None
        for i, layer in enumerate(self.pc.layers):
            num_activations = layer.num_features_out * layer.num_sums_out * layer.num_repetitions

            if num_activations == self.latent_dim:
                self.layer_extraction_idx = i
                break

        # HACK: Add this property to the Einet to extract activations in sampling_contex_act
        setattr(self.pc, "layer_extraction_idx", self.layer_extraction_idx)

        if self.layer_extraction_idx is None:
            raise ValueError(f"Could not find a layer with num_features_out == latent_dim ({self.latent_dim}).")

        # Add BatchNorm1D
        if self.cfg.apc.decoder == DecoderType.NN:
            self.bn = nn.BatchNorm1d(self.latent_dim)

    def _make_pc(self) -> Einet:
        leaf_kwargs, leaf_type = get_leaf_args(cfg=self.cfg)

        config = EinetConfig(
            num_features=self.data_shape.num_pixels,
            num_channels=self.data_shape.channels,
            depth=self.cfg.einet.D,
            num_sums=self.cfg.einet.S,
            num_leaves=self.cfg.einet.I,
            num_repetitions=self.cfg.einet.R,
            num_classes=1,
            leaf_type=leaf_type,
            leaf_kwargs=leaf_kwargs,
            layer_type=self.cfg.einet.layer,
            dropout=0.0,
            structure=self.cfg.einet.structure,
        )
        einet = EinetSpnae(config, latent_dim=self.latent_dim)

        return einet

    def encode(
            self, x: torch.Tensor, mpe=True, mpe_at_leaves=False, return_leaf_params=False, tau=1.0) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input data.

        Args:
            x: Input data.
            mpe: If True, performs MPE encoding.
            mpe_at_leaves: If True, performs MPE encoding at the leaves.
            return_leaf_params: If True, returns leaf parameters.
            tau: Temperature for sampling.

        Returns:
            torch.Tensor: Encoded data.
        """

        # Pick encoding method
        match self.cfg.apc.encoder:
            case PcEncoderType.EINET_ACT:
                encoder = self._encode_act
            case PcEncoderType.EINET_CAT:
                encoder = self._encode_cat
            case _:
                raise ValueError(f"Unknown encoder {self.cfg.apc.encoder}.")

        return encoder(x, mpe=mpe, mpe_at_leaves=mpe_at_leaves, return_leaf_params=return_leaf_params, tau=tau)

    def _encode_act(
        self, x: torch.Tensor, mpe: bool = True, mpe_at_leaves: bool = True, return_leaf_params: bool = False, tau=1.0
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Modified forward pass of the Einet to extract intermediate activations."""
        if x.dim() == 2:  # [N, D]
            x = x.unsqueeze(1)

        if x.dim() == 4:  # [N, C, H, W]
            x = x.view(x.shape[0], self.pc.config.num_channels, -1)

        assert x.dim() == 3
        assert (
            x.shape[1] == self.pc.config.num_channels
        ), f"Number of channels in input ({x.shape[1]}) does not match number of channels specified in config ({self.pc.config.num_channels})."
        assert (
            x.shape[2] == self.pc.config.num_features
        ), f"Number of features in input ({x.shape[0]}) does not match number of features specified in config ({self.pc.config.num_features})."

        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        x = self.pc.leaf(x, marginalized_scopes=None)

        # Pass through intermediate layers
        for i, layer in enumerate(self.pc.layers):
            x = layer(x)

            if i == self.layer_extraction_idx:
                # Activation encoding: simply take the activations of the current layer
                z = x
                break

        if self.cfg.apc.decoder == DecoderType.NN and self.cfg.apc.encoder == PcEncoderType.EINET_ACT:
            # Apply batchnorm to normalize activations only for ACT encoder and NN decoder
            shape = z.shape
            z = self.bn(z.view(z.shape[0], self.latent_dim))
            z = z.view(shape)

        assert z is not None

        if return_leaf_params:
            return None, None, z
        else:
            return z


    def _encode_cat(
            self, evidence: torch.Tensor, mpe=True, mpe_at_leaves=False, return_leaf_params=False, tau=1.0) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        num_samples = evidence.shape[0]

        indices_out = torch.ones(size=(num_samples, 1, 1), dtype=torch.float, device=self.pc._device, requires_grad=True)
        indices_repetition = torch.ones(size=(num_samples, 1), dtype=torch.float, device=self.pc._device, requires_grad=True)

        ctx = SamplingContext(
            num_samples=num_samples,
            is_mpe=mpe,
            mpe_at_leaves=mpe_at_leaves,
            temperature_leaves=1.0,
            temperature_sums=1.0,
            num_repetitions=self.pc.config.num_repetitions,
            evidence=evidence,
            indices_out=indices_out,
            indices_repetition=indices_repetition,
            is_differentiable=True,
            return_leaf_params=return_leaf_params,
        )
        with sampling_context_act_einet(model=self.pc, evidence_data=evidence, requires_grad=True, seed=self.cfg.seed):
            # Save parent indices that were sampled from the sampling root
            if self.pc.config.num_repetitions > 1:
                indices_out_pre_root = ctx.indices_out
                ctx = self.pc.mixing.sample(ctx=ctx)

                # Obtain repetition indices
                ctx.indices_repetition = ctx.indices_out.view(num_samples, self.pc.config.num_repetitions)
                ctx.indices_out = indices_out_pre_root

            # Sample inner layers in reverse order (starting from topmost)
            for i, layer in reversed(list(enumerate(self.pc.layers))):
                if i == self.layer_extraction_idx:

                    if i == len(self.pc.layers) - 1 and self.pc.config.num_repetitions > 1:
                        # Encoding is from the layer below the repetition mixing layer
                        z = ctx.indices_repetition
                    else:
                        # CAT encoding
                        z = ctx.indices_out
                    break

                ctx = layer.sample(ctx=ctx)

        if return_leaf_params:
            return None, None, z
        else:
            return z

    def decode(self, z: torch.Tensor, mpe = True, mpe_at_leaves = True, return_leaf_params = False, tau = 1.0, fill_evidence = False) -> torch.Tensor:
        # Sample from PC
        x_dec = _pc_sample(
            pc=self.pc,
            z=z,
            is_differentiable=True,
            is_mpe=mpe,
            mpe_at_leaves=mpe,
            temperature_leaves=tau,
            temperature_sums=tau,
            cfg=self.cfg,
            fill_evidence=fill_evidence,
        )
        return x_dec.view(z.shape[0], *self.data_shape)

    def log_likelihood(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.pc(x)

    def sample_z(
        self,
        num_samples: int,
        seed: int,
        return_leaf_params: bool = False,
        tau: float = 1.0,
    ) -> torch.Tensor:
        # Sample x and then encode
        x = _pc_sample(
            pc=self.pc,
            num_samples=num_samples,
            seed=seed,
            is_differentiable=False,
            return_leaf_params=return_leaf_params,
            cfg=self.cfg,
        )

        return self.encode(x, return_leaf_params=return_leaf_params, tau=tau)
