import logging
from contextlib import contextmanager, nullcontext
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch import nn

from apc.models.conv_pc import SumProdLayer, SumLayer, SamplingContext, ProdConv
from apc.models.encoder.abstract_encoder import AbstractEncoder
from apc.models.encoder.conv_pc_encoder import sampling_context
from apc.enums import PcEncoderType, DecoderType
from simple_einet.abstract_layers import logits_to_log_weights
from simple_einet.layers.distributions.binomial import Binomial
from simple_einet.sampling_utils import SIMPLE
from simple_einet.sampling_utils import index_one_hot

logger = logging.getLogger(__name__)


@contextmanager
def sampling_context_act(
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
            _ = model.log_likelihood(x=evidence_data)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        if evidence_data is not None:
            model.layers[model.layer_extraction_idx]._disable_input_cache()

    if seed is not None:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)


class ConvPcSpnAeEncoder(AbstractEncoder):
    """
    This class implements ACT and CAT encoder from the SPNAE paper [1].

    The model makes use of a simple convolutional SPN architecture with alternating sum and product layers.

    Each product layer reduces the height and width by a factor of 2, while the sum layers keep the height and width constant but map the number of nodes per scope from in_channels to out_channels.

    [1] Vergari, Antonio et al. “Sum-Product Autoencoding: Encoding and Decoding Representations Using Sum-Product Networks.” AAAI Conference on Artificial Intelligence (2018).
    """

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)
        self.channels = cfg.conv_pc.channels
        self.kernel_size = cfg.conv_pc.kernel_size
        self.depth = cfg.conv_pc.depth
        self.latent_dim = cfg.latent_dim

        # Data leaf
        self.leaf_data = Binomial(
            num_features=self.data_shape.num_pixels,
            num_channels=self.data_shape.channels,
            num_leaves=self.channels,
            num_repetitions=1,
            total_count=2**cfg.n_bits - 1,
        )
        self.latent_depth = self.cfg.conv_pc.latent_depth
        self.latent_channels = self.cfg.conv_pc.latent_channels

        # Construct layers
        layers = []
        h, w = self.data_shape.height, self.data_shape.width

        kh, kw = self.kernel_size, self.kernel_size

        if self.cfg.conv_pc.split == "bottom-up":
            for i in range(0, self.depth - 1):
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

                # Update height and width according to the kernel size
                h, w = h // kh, w // kw

            # Reduce to a single dimension by setting the kernel size the the current height and width
            # Add a root sum layer to reduce to 1 channel (single root node)
            layers.append(
                SumProdLayer(
                    order="prod-sum",
                    in_channels=self.channels,
                    out_channels=1,
                    height=h,
                    width=w,
                    kernel_size_h=h,
                    kernel_size_w=w,
                    sum_conv=cfg.conv_pc.sum_conv,
                )
            )

        elif self.cfg.conv_pc.split == "top-down":

            # Add sum layer on top
            layers.append(SumLayer(in_channels=self.channels, out_channels=1, height=1, width=1))

            # Build from top down
            h, w = 1, 1
            for i in reversed(range(0, self.depth - 1)):
                h, w = h * 2, w * 2
                layers.append(
                    SumProdLayer(
                        in_channels=self.channels,
                        out_channels=self.channels,
                        height=h,
                        width=w,
                        kernel_size_h=2,
                        kernel_size_w=2,
                        order=cfg.conv_pc.order,
                        sum_conv=cfg.conv_pc.sum_conv,
                    )
                )

            # Add lowest SumProd layer to reduce data height/width to h/w
            layers.append(
                ProdConv(
                    kernel_size_h=self.data_shape.height // h,
                    kernel_size_w=self.data_shape.width // w,
                )
            )
            layers = reversed(layers)

        else:
            raise ValueError(f"Split {self.cfg.conv_pc.split} not supported")

        self.layers = nn.ModuleList(layers)

        if self.cfg.apc.decoder == DecoderType.NN:
            # Add BatchNorm
            self.bn = nn.BatchNorm1d(self.latent_dim)

        # We find the layer activation extraction index by going through all layers and finding the layer with the num_features_out that matches the latent_dim
        self.layer_extraction_idx = None
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SumProdLayer):
                num_activations = layer.num_features_out * layer.out_channels

                if num_activations == self.latent_dim:
                    self.layer_extraction_idx = i
                    break

        if self.layer_extraction_idx is None:
            raise ValueError(f"Could not find a layer with num_features_out == latent_dim ({self.latent_dim}).")

    @property
    def __device(self):
        """Small hack to obtain the current device."""
        return next(self.parameters()).device

    def _forward_leaf_data(self, x: torch.Tensor):
        """
        Forward data through leaf distributions.

        Args:
            x: Input data.

        Returns:
            torch.Tensor: Leaf log-likehoods.
        """
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
        """
        Compute the log-likelihood of the given data. Ignores z.

        Args:
            x: Input data.
            z: Latent data (ignored).

        Returns:
            torch.Tensor: Log-likelihood of the given data.
        """

        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        x = self._forward_leaf_data(x)

        for layer in self.layers:
            x = layer(x)

        return x.view(x.size(0))

    def sample(
        self,
        num_samples: int = None,
        evidence_data: torch.Tensor | None = None,
        evidence_latents: torch.Tensor | None = None,
        seed: int | None = None,
        mpe: bool = False,
        return_data: bool = True,
        return_latents: bool = False,
        return_leaf_params: bool = False,
        is_differentiable: bool = True,
        tau=1.0,
        fill_evidence=False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the model.

        Args:
            num_samples: Number of samples to generate.
            evidence_data: Evidence data.
            evidence_latents: Evidence latents.
            seed: Seed for sampling.
            mpe: If True, performs MPE sampling.
            return_data: If True, returns the generated data.
            return_latents: If True, returns the generated latents.
            return_leaf_params: If True, returns leaf parameters.
            is_differentiable: If True, uses differentiable sampling.
            tau: Temperature for sampling.
            fill_evidence: If True, fills the evidence data with the generated data.

        Returns:
            torch.Tensor: Generated data.

        """

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
            is_differentiable=is_differentiable,
            return_leaf_params=return_leaf_params,
            is_mpe=mpe,
            temperature_leaves=tau,
            temperature_sums=tau,
        )

        # Init indices_out: (N, C=1, H=1, W=1)
        ctx.indices_out = torch.ones(
            size=(num_samples, 1, 1, 1), dtype=torch.float, device=self.__device, requires_grad=True
        )

        # Only really relevant for the leaf layer. Intermediate layers only have diff-sampling impelemented for now
        if is_differentiable:
            ctx.indices_repetition = torch.ones(
                size=(num_samples, 1), dtype=torch.float, device=self.__device, requires_grad=True
            )
        else:
            ctx.indices_repetition = torch.zeros(
                size=(num_samples, 1), dtype=torch.long, device=self.__device, requires_grad=False
            )


        has_latents = evidence_latents is not None

        if has_latents and self.cfg.apc.encoder == PcEncoderType.CONV_PC_SPNAE_ACT:
            # Set evidence as input cache for the sum layer above
            activated_layer = self.layers[self.layer_extraction_idx + 1]
            if isinstance(activated_layer, SumProdLayer):
                activated_layer = activated_layer.sum_layer
            activated_layer._enable_input_cache()
            activated_layer._input_cache["x"] = evidence_latents

        # Iterate over layers
        for i, layer in reversed(list(enumerate(self.layers))):
            if self.cfg.apc.encoder == PcEncoderType.CONV_PC_SPNAE_ACT:
                # Sample layer
                ctx = layer.sample(ctx=ctx)
            else:
                if has_latents and i == self.layer_extraction_idx:
                    # Insert CAT encoding
                    ctx.indices_out = evidence_latents

                # Sample layer
                ctx = layer.sample(ctx=ctx)

        if has_latents and self.cfg.apc.encoder == PcEncoderType.CONV_PC_SPNAE_ACT:
            activated_layer._disable_input_cache()

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
            self, x: torch.Tensor, mpe=True, mpe_at_leaves=True, return_leaf_params=False, tau=1.0) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Convert to match statement
        match self.cfg.apc.encoder:
            case PcEncoderType.CONV_PC_SPNAE_ACT:
                encoder = self._encode_act
            case PcEncoderType.CONV_PC_SPNAE_CAT:
                encoder = self._encode_cat
            case _:
                raise ValueError(f"Unknown encoder {self.cfg.apc.encoder}.")


        return encoder(x, mpe=mpe, mpe_at_leaves=mpe_at_leaves, return_leaf_params=return_leaf_params, tau=tau)



    def _encode_act(
        self, x: torch.Tensor, mpe=True, mpe_at_leaves=True, return_leaf_params=False, tau=1.0) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input data using the ACT encoder from the SPNAE paper.

        Args:
            x: Input data.
            mpe: If True, performs MPE encoding.
            mpe_at_leaves: If True, performs MPE encoding at the leaves.
            return_leaf_params: If True, returns leaf parameters.
            tau: Temperature for sampling.

        Returns:
            torch.Tensor: Encoded data.

        """

        # Apply leaf distributions (replace marginalization indicators with 0.0 first)
        x = self._forward_leaf_data(x)

        # Pass through intermediate layers
        for i, layer in enumerate(self.layers):
            # ACT encoder
            x = layer(x)

            if i == self.layer_extraction_idx:
                # Activation encoding: simply take the activations of the current layer
                z = x
                break

        if self.cfg.apc.decoder == DecoderType.NN and self.cfg.apc.encoder == PcEncoderType.CONV_PC_SPNAE_ACT:
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
            self, x: torch.Tensor, mpe=True, mpe_at_leaves=False, return_leaf_params=False, tau=1.0) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input data using the CAT encoder from the SPNAE paper.

        Args:
            x: Input data.
            mpe: If True, performs MPE encoding.
            mpe_at_leaves: If True, performs MPE encoding at the leaves.
            return_leaf_params: If True, returns leaf parameters.
            tau: Temperature for sampling.

        Returns:
            torch.Tensor: Encoded data.
        """

        num_samples = x.shape[0]

        ctx = SamplingContext(
            num_samples=num_samples,
            mpe_at_leaves=mpe,
            is_differentiable=True,
            return_leaf_params=return_leaf_params,
            is_mpe=mpe,
            temperature_leaves=tau,
            temperature_sums=tau,
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
                evidence_data=x,
                evidence_latents=None,
                requires_grad=True,
                seed=self.cfg.seed,
                enable_input_cache=True
        ):
            # Iterate over layers
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == self.layer_extraction_idx:
                    # CAT encoding
                    z = ctx.indices_out
                    break

                # Sample layer
                ctx = layer.sample(ctx=ctx)

        if return_leaf_params:
            return None, None, z
        else:
            return z

    def decode(
        self,
        z: torch.Tensor,
        mpe=True,
        mpe_at_leaves=True,
        return_leaf_params=False,
        tau=1.0,
        fill_evidence=False,
    ) -> torch.Tensor:
        """
        Decode embeddings.

        Args:
            z: Latent data.
            mpe: If True, performs MPE decoding.
            mpe_at_leaves: If True, performs MPE decoding at the leaves.
            return_leaf_params: If True, returns leaf parameters.
            tau: Temperature for sampling.
            fill_evidence: If True, fills the evidence data with the generated data.

        Returns:
            torch.Tensor: Decoded data.
        """

        # Sample from PC
        x = self.sample(
            evidence_data=None,
            evidence_latents=z,
            mpe=mpe,
            return_data=True,
            return_leaf_params=return_leaf_params,
            tau=tau,
            fill_evidence=fill_evidence,
        )
        return x.view(-1, *self.data_shape)

    def sample_z(self, num_samples: int, seed: int, return_leaf_params=False, tau=1.0) -> torch.Tensor:
        """
        We can sample a new embedding by sampling a new data point and encoding it.

        Args:
            num_samples: Number of samples.
            seed: Seed for sampling.
            return_leaf_params: If True, returns leaf parameters.
            tau: Temperature for sampling.

        Returns:
            torch.Tensor: Sampled embeddings.

        """
        x = self.sample(
            num_samples=num_samples, seed=seed, return_leaf_params=return_leaf_params, is_differentiable=False, tau=tau
        )
        return self.encode(x, return_leaf_params=return_leaf_params, tau=tau)
