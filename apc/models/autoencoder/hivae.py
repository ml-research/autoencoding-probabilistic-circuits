# #!/usr/bin/env python3

from apc.models.decoder.abstract_decoder import AbstractDecoder
from apc.models.autoencoder.variational_autoencoder import VariationalAutoencoder
import warnings
from apc.enums import DecoderType, ModelName
from apc.models.components import ResidualStack
from apc.models.encoder.abstract_encoder import AbstractEncoder
from apc.models.decoder.nn_decoder import NeuralDecoder1D, NeuralDecoder2D
from simple_einet.data import get_data_shape, is_1d_data
from apc.models.encoder.nn_encoder import NeuralEncoder1D, NeuralEncoder2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from omegaconf import DictConfig, OmegaConf  # Assuming OmegaConf is used for cfg
from typing import List, Dict, Tuple, Any, Optional

from apc.models.abstract_model import AbstractAutoencoder
from apc.exp_utils import seed_context  # For sample_z

from apc.models.encoder.nn_encoder import DummyEncoder
from apc.models.decoder.nn_decoder import DummyDecoder
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical
import math


# def to_one_hot(x, size):
#     x_one_hot = x.new_zeros(x.size(0), size)
#     x_one_hot.scatter_(1, x.unsqueeze(-1).long(), 1).float()

#     return x_one_hot


# class GumbelDistribution(ExpRelaxedCategorical):
#     @torch.no_grad()
#     def sample(self, sample_shape=torch.Size()):
#         return OneHotCategorical(probs=self.probs).sample(sample_shape)

#     def rsample(self, sample_shape=torch.Size()):
#         return torch.exp(super().rsample(sample_shape))

#     @property
#     def mean(self):
#         return self.probs

#     def expand(self, batch_shape, _instance=None):
#         return super().expand(batch_shape[:-1], _instance)

#     def log_prob(self, value):
#         return OneHotCategorical(probs=self.probs).log_prob(value)


# class NeuralEncoder1DHiVAE(AbstractEncoder):
#     """
#     Modification of the original NeuralEncoder1D to support HiVAE s latents as input concatenation to x.
#     """

#     def __init__(
#         self,
#         cfg: DictConfig,
#     ):
#         super().__init__(cfg=cfg)
#         hidden_dim = cfg.nn_encoder.num_hidden
#         act_fn = lambda: nn.LeakyReLU(0.1)
#         self.data_shape = get_data_shape(cfg.dataset)

#         self.last_layer_dim = self.cfg.latent_dim if self.cfg.model_name == ModelName.AE else self.cfg.latent_dim * 2

#         input_dim = (
#             self.data_shape.height + self.cfg.hivae.L
#         )  # Add L to input dimension as this encoder gets x and s as input with |s| = L

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim * 16),
#             act_fn(),
#             nn.Linear(hidden_dim * 16, hidden_dim * 8),
#             act_fn(),
#             nn.Linear(hidden_dim * 8, hidden_dim * 4),
#             act_fn(),
#             nn.Linear(hidden_dim * 4, hidden_dim * 2),
#             act_fn(),
#             nn.Linear(hidden_dim * 2, self.last_layer_dim),
#         )

#     def encode(self, x, s: torch.Tensor, mpe: bool = True, tau=1.0):
#         mask = torch.isnan(x)
#         x[mask] = self.cfg.nan_mask_value
#         x = x.view(x.shape[0], self.data_shape.num_pixels)  # Flatten the input
#         # Concatenate s to x
#         x = torch.cat((x, s), dim=1)
#         x = self.net(x)
#         x = x.view(x.shape[0], self.last_layer_dim)
#         return x


# class NeuralEncoder2DHiVAE(AbstractEncoder):
#     """
#     Modification of the original NeuralEncoder2D to support HiVAE s latents as input concatenation to deep representation of x.
#     """

#     def __init__(
#         self,
#         cfg: DictConfig,
#         latent_dim: int,
#         has_s: bool = True,
#     ):
#         super().__init__(cfg)

#         self.has_s = has_s  # Gets (x, s) as input

#         h, w = self.data_shape.height, self.data_shape.width
#         in_channels = self.data_shape.channels

#         num_hidden = cfg.nn_encoder.num_hidden
#         num_residual_layers = cfg.nn_encoder.num_res_layers
#         num_residual_hidden = cfg.nn_encoder.num_res_hidden

#         self._conv_1 = nn.Conv2d(
#             in_channels=in_channels, out_channels=num_hidden // 2, kernel_size=4, stride=2, padding=1
#         )
#         self._conv_2 = nn.Conv2d(
#             in_channels=num_hidden // 2, out_channels=num_hidden, kernel_size=4, stride=2, padding=1
#         )
#         assert (
#             self.cfg.nn_encoder.num_scales >= 2
#         ), f"Number of scales should be equal or greater than 2 but was {self.cfg.nn_encoder.num_scales}"
#         layers = []
#         for i in range(self.cfg.nn_encoder.num_scales - 2):
#             layers.append(
#                 nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=4, stride=2, padding=1)
#             )
#         self.scales = nn.ModuleList(layers)

#         self._conv_3 = nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, stride=1, padding=1)

#         self._residual_stack = ResidualStack(
#             in_channels=num_hidden,
#             num_hiddens=num_hidden,
#             num_residual_layers=num_residual_layers,
#             num_residual_hiddens=num_residual_hidden,
#             bn=cfg.nn_encoder.bn,
#         )

#         linear_in = h // 2**self.cfg.nn_encoder.num_scales * w // 2**self.cfg.nn_encoder.num_scales * num_hidden
#         linear_in = linear_in

#         linear_out = latent_dim

#         if self.has_s:
#             self._conv_4 = nn.Conv2d(
#                 in_channels=num_hidden, out_channels=num_hidden // 2, kernel_size=3, stride=1, padding=1
#             )
#             linear_in = linear_in // 2
#             linear_out = linear_out * 2

#             linear_in = (
#                 linear_in + self.cfg.hivae.L
#             )  # Add L to input dimension as this encoder gets x and s as input with |s| = L

#         if linear_in <= linear_out:
#             warnings.warn(
#                 f"Encoder linear layer should reduce the dimensionality but got Linear(in_features={linear_in}, out_features={linear_out})."
#             )

#         self.linear = nn.Linear(linear_in, linear_out)

#     def encode(self, x: torch.Tensor, s: torch.Tensor = None, mpe: bool = True, tau=1.0) -> torch.Tensor:
#         x = x.view(x.shape[0], self.data_shape.channels, self.data_shape.height, self.data_shape.width)

#         x = self._conv_1(x)
#         x = F.relu(x)

#         x = self._conv_2(x)
#         x = F.relu(x)

#         for scale in self.scales:
#             x = scale(x)
#             x = F.relu(x)

#         x = self._conv_3(x)
#         x = self._residual_stack(x)

#         if self.has_s:
#             x = self._conv_4(x)

#         # Flatten
#         x = x.view(x.size(0), -1)

#         # Concatenate s to x
#         if s is not None:
#             x = torch.cat((x, s), dim=1)
#         x = self.linear(x)

#         return x


# class NeuralDecoder2DHiVAE(AbstractDecoder):

#     def __init__(
#         self,
#         cfg: DictConfig,
#         latent_dim: int,
#         has_s: bool = True,
#     ):
#         super().__init__(cfg)
#         self.has_s = has_s

#         h, w = self.data_shape.height, self.data_shape.width

#         num_hidden = cfg.nn_decoder.num_hidden
#         num_res_layers = cfg.nn_decoder.num_res_layers
#         num_res_hidden = cfg.nn_decoder.num_res_hidden

#         self.first_h = h // 2**self.cfg.nn_decoder.num_scales
#         self.first_w = w // 2**self.cfg.nn_decoder.num_scales

#         if has_s:
#             latent_dim = latent_dim + cfg.hivae.L

#         self.latent_dim = latent_dim

#         self.linear = nn.Linear(latent_dim, self.first_h * self.first_w * num_hidden)

#         self._conv_1 = nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, stride=1, padding=1)

#         self._residual_stack = ResidualStack(
#             in_channels=num_hidden,
#             num_hiddens=num_hidden,
#             num_residual_layers=num_res_layers,
#             num_residual_hiddens=num_res_hidden,
#             bn=cfg.nn_decoder.bn,
#         )
#         assert self.cfg.nn_decoder.num_scales >= 2, "Number of scales must be greater than or equal to 2"

#         layers = []
#         for i in range(self.cfg.nn_decoder.num_scales - 2):
#             layers.append(
#                 nn.ConvTranspose2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=4, stride=2, padding=1)
#             )
#         self.scales = nn.ModuleList(layers)

#         self._conv_trans_1 = nn.ConvTranspose2d(
#             in_channels=num_hidden, out_channels=num_hidden // 2, kernel_size=4, stride=2, padding=1
#         )

#         self._conv_trans_2 = nn.ConvTranspose2d(
#             in_channels=num_hidden // 2, out_channels=self.data_shape.channels, kernel_size=4, stride=2, padding=1
#         )

#     def decode(self, z: torch.Tensor, s: torch.Tensor=None, tau=1.0) -> torch.Tensor:
#         z = z.view(z.size(0), self.cfg.latent_dim)

#         if self.has_s:
#             # Concatenate s to z
#             z = torch.cat((z, s), dim=1)

#         # Linear layer
#         x = self.linear(z)
#         x = x.view(x.size(0), -1, self.first_h, self.first_w)

#         x = self._conv_1(x)

#         x = self._residual_stack(x)

#         for scale in self.scales:
#             x = scale(x)
#             x = F.relu(x)

#         x = self._conv_trans_1(x)
#         x = F.relu(x)

#         x = self._conv_trans_2(x)

#         # if self.cfg.model_name not in [ModelName.VAEM]:
#         if self.cfg.model_name not in [ModelName.VAEM, ModelName.MIWAE]:
#             x = torch.tanh(x)

#         x = x.view(x.shape[0], *self.data_shape)

#         return x

#     def forward(self, z: torch.Tensor, s: torch.Tensor = None, tau=1.0) -> torch.Tensor:
#         return self.decode(z, s=s, tau=tau)


# class HIVAE(VariationalAutoencoder):
#     def __init__(self, cfg: DictConfig):
#         super().__init__(cfg)

#         self.L = self.cfg.hivae.L

#         if is_1d_data(self.cfg.dataset):
#             # Encoder x -> s
#             self.encoder_s = NeuralEncoder1DHiVAE(self.cfg)
#             # Encoder (x, s) -> z
#             self.encoder_z = NeuralEncoder1DHiVAE(self.cfg)

#             self.decoder = NeuralDecoder1D(self.cfg)
#         else:
#             # Encoder x -> s
#             self.encoder_s = NeuralEncoder2DHiVAE(self.cfg, has_s=False, latent_dim=self.L)
#             # Encoder (x, s) -> z
#             self.encoder_z = NeuralEncoder2DHiVAE(self.cfg, has_s=True, latent_dim=self.latent_dim)

#             self.decoder = NeuralDecoder2DHiVAE(self.cfg, has_s=True, latent_dim=self.latent_dim)

#         self.p_z_loc = nn.Linear(self.L, self.latent_dim)

#     def make_encoder(self) -> DummyEncoder:
#         return DummyEncoder(self.cfg)

#     def make_decoder(self) -> NeuralDecoder1D | NeuralDecoder2D:
#         if is_1d_data(self.cfg.dataset):
#             return NeuralDecoder1D(self.cfg)
#         else:
#             return NeuralDecoder2D(self.cfg)

#     def encode(self, x: torch.Tensor, mpe: bool = True, tau: float = 1.0, loss_call=False) -> torch.Tensor:
#         # Encode x to s
#         s_logits = self.encoder_s.encode(x, s=None, mpe=mpe, tau=tau).log_softmax(dim=-1)

#         if mpe:
#             s_samples = to_one_hot(torch.argmax(s_logits, dim=-1), s_logits.size(-1)).float()
#         else:
#             gumbel = GumbelDistribution(temperature=tau, logits=s_logits)
#             s_samples = gumbel.rsample() if self.training else gumbel.sample()

#         # Encode (x, s) to z
#         z_loc, z_logvar = self.encoder_z.encode(x, s=s_samples, mpe=mpe, tau=tau).chunk(2, dim=1)

#         if mpe:
#             return z_loc

#         q_z = Normal(loc=z_loc, scale=torch.exp(z_logvar / 2))
#         z_samples = q_z.rsample()

#         if loss_call:
#             return z_samples, q_z, s_samples, s_logits

#         return z_samples

#     def decode(self, z: torch.Tensor, s: torch.Tensor = None, decoder_type: DecoderType = None, tau: float = 1.0) -> torch.Tensor:
#         if s is None:
#             s = self.prior_s.sample((z.shape[0],))
#         y_shared = self.decoder(z, s=s)

#         # Flatten once more
#         x_rec = y_shared.view(y_shared.shape[0], -1)
#         return x_rec.view(x_rec.shape[0], *self.data_shape)

#     @property
#     def prior_s(self):
#         """Prior distribution for s."""
#         return OneHotCategorical(probs=torch.ones(self.L, device=self.mydevice) / self.L, validate_args=False)

#     def prior_z(self, loc):
#         """Prior distribution for z."""
#         return Normal(loc=loc, scale=1.0)

#     def loss(self, x) -> dict[str, torch.Tensor]:

#         # Get nan mask and replace with nan_mask_value
#         mask = torch.isnan(x)
#         if mask.any():
#             x[mask] = 0.0

#         # Encode x to s
#         z, q_z, s_samples, s_logits = self.encode(x, mpe=False, loss_call=True)

#         # KL-Div s
#         log_qs_x = OneHotCategorical(logits=s_logits, validate_args=False).log_prob(s_samples)
#         log_ps = self.prior_s.log_prob(s_samples)
#         kl_s = log_qs_x - log_ps

#         # KL-Div z
#         pz_loc = self.p_z_loc(s_samples)
#         log_qz_x = q_z.log_prob(z).sum(-1)
#         log_pz = self.prior_z(pz_loc).log_prob(z).sum(-1)
#         kl_z = log_qz_x - log_pz

#         kl = kl_s + kl_z

#         # Decode
#         x_rec = self.decode(z, s=s_samples)

#         if self.cfg.train.rec_loss == "mse":
#             loss_rec = F.mse_loss(x_rec, x, reduction="sum") / x.shape[0]
#         elif self.cfg.train.rec_loss == "bce":
#             loss_rec = F.binary_cross_entropy(x_rec, x, reduction="sum") / x.shape[0]
#         else:
#             raise ValueError(f"Unknown rec_loss: {self.cfg.train.rec_loss}")

#         return {
#             "rec": loss_rec,
#             "kld": kl.mean(),
#         }


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.distributions.kl import kl_divergence
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Any, Optional
import warnings

# Gumbel Distribution Helper
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical

def to_one_hot(x, size):
    """Converts a tensor of indices to a one-hot representation."""
    x_one_hot = x.new_zeros(x.size(0), size)
    # Ensure indices are long type for scatter_
    x_one_hot.scatter_(1, x.unsqueeze(-1).long(), 1).float()
    return x_one_hot

class GumbelDistribution(ExpRelaxedCategorical):
    """Gumbel-Softmax distribution supporting rsample and hard sampling."""
    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        # Sample hard one-hot vector for evaluation/MPE
        return OneHotCategorical(probs=self.probs).sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        # Sample soft, differentiable vector using Gumbel-Softmax trick
        # Note: The original HI-VAE code uses exp(super().rsample()), which gives RelaxedOneHotCategorical samples.
        # For straight-through, we need to implement it or use a library function if available.
        # Let's stick to the Relaxed version for differentiability, common in VAEs.
        # If hard samples are strictly needed *during training* for loss, use straight-through.
        # For now, rsample returns the standard RelaxedOneHotCategorical sample.
        # Update: The original code *did* use exp(super().rsample()), let's keep it.
        return torch.exp(super().rsample(sample_shape))

    def rsample_hard_straight_through(self, sample_shape=torch.Size()):
        """ Samples hard one-hot vector using straight-through estimator. """
        soft_sample = self.rsample(sample_shape) # Get differentiable sample
        hard_sample = to_one_hot(torch.argmax(soft_sample, dim=-1), self.logits.shape[-1]).float()
        return (hard_sample - soft_sample).detach() + soft_sample # Straight-through

    @property
    def mean(self):
        return self.probs

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GumbelDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        logits_shape = batch_shape + self.logits.shape[-1:]
        new.temperature = self.temperature
        new.logits = self.logits.expand(logits_shape)
        # Ensure base class init is called correctly
        super(ExpRelaxedCategorical, new).__init__(batch_shape, validate_args=False)
        # Manually expand probs if needed by base class logic (may not be necessary)
        if hasattr(self, '_probs'):
             new._probs = self._probs.expand(logits_shape)
        return new

    def log_prob(self, value):
        # Log prob of a hard one-hot vector under the base categorical distribution
        return OneHotCategorical(probs=self.probs).log_prob(value)


class HIVAE(VariationalAutoencoder):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data_shape = get_data_shape(cfg.dataset)
        # Ensure hivae config exists
        if 'hivae' not in self.cfg:
             raise ValueError("HIVAE configuration ('hivae') missing in the main config.")
        self.L = self.cfg.hivae.L
        self.latent_dim = self.cfg.latent_dim # z dim from parent VAE config
        # self.y_dim_per_attribute = self.cfg.hivae.y_dim_per_attribute
        self.y_dim_per_attribute = 5

        if is_1d_data(self.cfg.dataset):
            self.total_input_dim = self.data_shape.height
        else:
            self.total_input_dim = self.data_shape.channels * self.data_shape.height * self.data_shape.width

        self.num_attributes = self.cfg.hivae.get("num_attributes", 1)
        if self.num_attributes == 1:
            warnings.warn("num_attributes set to 1. The intermediate Y layer will not be partitioned per attribute.")
            self.total_y_dim = self.y_dim_per_attribute
        else:
            self.total_y_dim = self.num_attributes * self.y_dim_per_attribute

        # Call parent __init__ AFTER setting attributes needed by make_encoder/decoder
        super().__init__(cfg)

        # --- Gumbel Temperature ---
        # Ensure hivae config is accessed correctly
        self.gumbel_temp = self.cfg.hivae.get("gumbel_temp_init", 1.0)
        self.gumbel_temp_anneal_rate = self.cfg.hivae.get('gumbel_temp_anneal_rate', None)
        self.gumbel_min_temp = self.cfg.hivae.get('gumbel_min_temp', 0.1)

        # --- KL Weights ---
        self.beta_s = self.cfg.hivae.get("beta_s", 1.0)
        self.beta_z = self.cfg.hivae.get("beta_z", 1.0)

        # Device handling - ensure model is moved to device after initialization
        _device_to_use = self.cfg.get("device", "cpu")
        self.to(torch.device(_device_to_use))
        self.device = self.mydevice


    def make_encoder(self) -> nn.Module:
        """Initializes HI-VAE specific encoder components."""
        # Ensure hivae config exists and access L correctly
        if 'hivae' not in self.cfg: raise ValueError("HIVAE config missing.")
        current_L = self.cfg.hivae.L

        # Encoder x -> s
        if is_1d_data(self.cfg.dataset):
            self.encoder_s = NeuralEncoder1D(self.cfg)
            # Modify last layer output dim to L
            if hasattr(self.encoder_s, 'net'): # Check if NeuralEncoder1D uses 'net'
                self.encoder_s.net[-1] = nn.Linear(self.encoder_s.net[-1].in_features, current_L)
            else: warnings.warn("Cannot modify output layer of encoder_s (1D). Assuming it's correct.")
        else:
            self.encoder_s = NeuralEncoder2D(self.cfg)
            if hasattr(self.encoder_s, 'linear'): # Check if NeuralEncoder2D uses 'linear'
                self.encoder_s.linear = nn.Linear(self.encoder_s.linear.in_features, current_L)
            else: warnings.warn("Cannot modify output layer of encoder_s (2D). Assuming it's correct.")


        # Encoder (x, s) -> z
        if is_1d_data(self.cfg.dataset):
            hidden_dim_z = self.cfg.nn_encoder.num_hidden
            act_fn_z = lambda: nn.LeakyReLU(0.1)
            input_dim_z = self.data_shape.height + current_L # x_dim + s_dim
            output_dim_z = self.latent_dim * 2 # mu and logvar for z
            self.encoder_z = nn.Sequential(
                nn.Linear(input_dim_z, hidden_dim_z * 16), act_fn_z(),
                nn.Linear(hidden_dim_z * 16, hidden_dim_z * 8), act_fn_z(),
                nn.Linear(hidden_dim_z * 8, hidden_dim_z * 4), act_fn_z(),
                nn.Linear(hidden_dim_z * 4, hidden_dim_z * 2), act_fn_z(),
                nn.Linear(hidden_dim_z * 2, output_dim_z),
            )
        else:
            # Need to instantiate NeuralEncoder2D correctly first
            temp_encoder_z_base = NeuralEncoder2D(self.cfg)
            # Calculate input size to its original linear layer
            h_final = self.data_shape.height // 2**self.cfg.nn_encoder.num_scales
            w_final = self.data_shape.width // 2**self.cfg.nn_encoder.num_scales
            num_hidden_final = self.cfg.nn_encoder.num_hidden
            # Check if VAE uses extra conv layer affecting linear layer input size
            # Use self.cfg.model_name if available
            model_name_enum = self.cfg.get("model_name", None) # Get model name if defined
            if model_name_enum in [ModelName.VAE, ModelName.VAEM, ModelName.MIWAE]:
                 num_hidden_final = num_hidden_final // 2

            linear_in_base = h_final * w_final * num_hidden_final
            linear_in_z = linear_in_base + current_L # Add s dim
            linear_out_z = self.latent_dim * 2 # mu and logvar for z

            # Replace the linear layer
            if hasattr(temp_encoder_z_base, 'linear'):
                temp_encoder_z_base.linear = nn.Linear(linear_in_z, linear_out_z)
            else: warnings.warn("Cannot modify linear layer of encoder_z (2D). Assuming it's correct.")
            self.encoder_z = temp_encoder_z_base

        # Return a dummy wrapper to satisfy abstract method signature
        class DummyEncoderWrapper(nn.Module):
            def __init__(self, s_net, z_net): super().__init__(); self.s_net = s_net; self.z_net = z_net
            def forward(self, x): return x
        return DummyEncoderWrapper(self.encoder_s, self.encoder_z)


    def make_decoder(self) -> nn.Module:
        """Initializes HI-VAE specific decoder components."""
        # Ensure hivae config exists
        if 'hivae' not in self.cfg: raise ValueError("HIVAE config missing.")
        hivae_cfg = self.cfg.hivae
        decoder_cfg = self.cfg.nn_decoder

        # 1. p(z|s) - Prior mean network
        self.p_z_loc_net = nn.Linear(self.L, self.latent_dim)

        # 2. g(z) -> Y network
        g_dec_layers = []
        g_input_dim = self.latent_dim
        g_hidden_dims = hivae_cfg.get("decoder_g_hidden_dims", [decoder_cfg.num_hidden * 2, decoder_cfg.num_hidden * 4])
        act_fn_g = lambda: nn.ReLU()

        current_dim_g = g_input_dim
        for h_dim in g_hidden_dims:
            g_dec_layers.append(nn.Linear(current_dim_g, h_dim))
            g_dec_layers.append(act_fn_g())
            current_dim_g = h_dim
        g_dec_layers.append(nn.Linear(current_dim_g, self.total_y_dim))
        self.g_decoder_net = nn.Sequential(*g_dec_layers)

        # 3. h_d(y_nd, s) -> intermediate representation heads
        self.h_decoder_heads = nn.ModuleList()
        intermediate_h_dim = hivae_cfg.get("intermediate_h_dim", 32)
        h_hidden_dims = hivae_cfg.get("decoder_h_attr_hidden_dims", [64])
        act_fn_h = lambda: nn.ReLU()

        for _ in range(self.num_attributes):
            h_layers = []
            h_input_dim = self.y_dim_per_attribute + self.L
            current_dim_h = h_input_dim
            for h_dim in h_hidden_dims:
                h_layers.append(nn.Linear(current_dim_h, h_dim))
                h_layers.append(act_fn_h())
                current_dim_h = h_dim
            h_layers.append(nn.Linear(current_dim_h, intermediate_h_dim))
            self.h_decoder_heads.append(nn.Sequential(*h_layers))

        # 4. Final layer(s) to map concatenated h_d outputs to x_rec
        final_input_dim = self.num_attributes * intermediate_h_dim
        self.final_decoder_layer = nn.Linear(final_input_dim, self.total_input_dim)

        # 5. Final activation
        if decoder_cfg.get("out_activation", "tanh") == "tanh":
            self.final_activation = nn.Tanh()
        elif decoder_cfg.get("out_activation", "tanh") == "sigmoid":
             self.final_activation = nn.Sigmoid()
        else: # Linear
             self.final_activation = nn.Identity()

        # Return a dummy wrapper
        class DummyDecoderWrapper(nn.Module):
            def __init__(self, g_net, h_heads, final_layer): super().__init__(); self.g_net=g_net; self.h_heads=h_heads; self.final_layer=final_layer
            def forward(self, x): return x
        return DummyDecoderWrapper(self.g_decoder_net, self.h_decoder_heads, self.final_decoder_layer)


    def encode(self, x: torch.Tensor, mpe: bool = True, tau: float = 1.0, loss_call=False) -> torch.Tensor:
        """ Encodes x -> s, then (x, s) -> z parameters. """
        x_no_nan = torch.nan_to_num(x, nan=self.cfg.get("nan_mask_value", 0.0))

        # --- Encode x -> s ---
        # Pass only x to encoder_s
        s_logits = self.encoder_s(x_no_nan) # No mpe/tau needed for base encoder call

        # --- Sample s ---
        if mpe:
            s_indices = torch.argmax(s_logits, dim=-1)
            s_samples = to_one_hot(s_indices, self.L).float().to(x.device)
        else:
            gumbel = GumbelDistribution(temperature=tau, logits=s_logits)
            # Use straight-through Gumbel for training loss if needed, else relaxed/hard sample
            # Let's use relaxed sample (rsample) for differentiability during loss calculation
            # If hard sample needed, use gumbel.rsample_hard_straight_through()
            s_samples = gumbel.rsample() # Differentiable soft sample

        # --- Encode (x, s) -> z ---
        if is_1d_data(self.cfg.dataset):
            x_flat = x_no_nan.view(x.shape[0], -1)
            z_encoder_input = torch.cat((x_flat, s_samples), dim=1)
            z_params = self.encoder_z(z_encoder_input) # Pass concatenated tensor
        else: # 2D
            # The NeuralEncoder2D needs adaptation to take concatenated input in its linear layer
            # Or we modify the call here. Let's modify the call:
            # 1. Get the convolutional features from encoder_z's base
            conv_features = self.encoder_z._conv_1(x_no_nan)
            conv_features = F.relu(conv_features)
            conv_features = self.encoder_z._conv_2(conv_features)
            conv_features = F.relu(conv_features)
            for scale in self.encoder_z.scales:
                conv_features = scale(conv_features)
                conv_features = F.relu(conv_features)
            conv_features = self.encoder_z._conv_3(conv_features)
            conv_features = self.encoder_z._residual_stack(conv_features)
            # Apply final conv if it exists (like in original VAE encoder)
            if hasattr(self.encoder_z, '_conv_4'):
                 conv_features = self.encoder_z._conv_4(conv_features)

            # 2. Flatten features and concatenate s
            conv_features_flat = conv_features.view(conv_features.size(0), -1)
            z_encoder_linear_input = torch.cat((conv_features_flat, s_samples), dim=1)

            # 3. Pass through the (modified) linear layer of encoder_z
            z_params = self.encoder_z.linear(z_encoder_linear_input)


        z_loc, z_logvar = z_params.chunk(2, dim=1)
        z_logvar = torch.clamp(z_logvar, -15.0, 15.0)

        # --- Return values ---
        if loss_call:
            q_z = Normal(loc=z_loc, scale=torch.exp(z_logvar / 2))
            z_samples_for_loss = q_z.rsample()
            return z_samples_for_loss, q_z, s_samples, s_logits
        else: # Inference call
            if mpe:
                return z_loc # Return mean
            else:
                q_z = Normal(loc=z_loc, scale=torch.exp(z_logvar / 2))
                return q_z.rsample() # Return sample


    def decode(self, z: torch.Tensor, s: torch.Tensor = None, decoder_type: DecoderType = None, tau: float = 1.0) -> torch.Tensor:
        """ Decodes latent variables z and s into reconstruction x_rec using the HI-VAE structure. """
        if s is None:
            prior_s_dist = OneHotCategorical(probs=torch.ones(self.L, device=z.device) / self.L)
            s = prior_s_dist.sample((z.shape[0],)).to(z.device)

        # 1. z -> Y
        Y_flat = self.g_decoder_net(z)

        # 2. Partition Y and process with h_d heads
        if self.num_attributes > 1:
             Y_partitioned = Y_flat.view(-1, self.num_attributes, self.y_dim_per_attribute)
        else:
             Y_partitioned = Y_flat.unsqueeze(1)

        h_outputs = []
        for i in range(self.num_attributes):
            y_ndi = Y_partitioned[:, i, :]
            h_input = torch.cat([y_ndi, s], dim=1)
            h_out = self.h_decoder_heads[i](h_input)
            h_outputs.append(h_out)

        # 3. Concatenate h_d outputs and pass through final layer
        h_concat = torch.cat(h_outputs, dim=1)
        x_rec_logits_or_values = self.final_decoder_layer(h_concat)

        # 4. Apply final activation
        x_rec = self.final_activation(x_rec_logits_or_values)

        # Reshape to original data shape
        return x_rec.view(x_rec.shape[0], *self.data_shape)


    @property
    def prior_s(self):
        """Prior distribution for s (uniform categorical)."""
        # Ensure device consistency
        return OneHotCategorical(probs=torch.ones(self.L, device=self.mydevice) / self.L, validate_args=False)

    def prior_z(self, s_samples: torch.Tensor):
        """Prior distribution for z given s: p(z|s) = N(loc=linear(s), scale=1.0)."""
        pz_loc = self.p_z_loc_net(s_samples)
        # Ensure scale is on the correct device
        scale = torch.ones_like(pz_loc, device=self.mydevice)
        return Normal(loc=pz_loc, scale=scale)


    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """ Computes the HI-VAE loss (Negative ELBO). """
        batch_size = x.shape[0]
        x_original = x.to(self.mydevice) # Keep original with NaNs

        # Encode: Get samples and distributions needed for loss
        # Use tau=self.gumbel_temp for sampling s during training loss calculation
        # Ensure loss_call=True to get all necessary return values
        # Note: encode expects potentially NaN input, handles it internally
        z_samples, q_z, s_samples, s_logits = self.encode(x_original, mpe=False, tau=self.gumbel_temp, loss_call=True)

        # --- KL Divergences ---
        # KL[q(s|x) || p(s)]
        # Use the GumbelDistribution's log_prob which delegates to OneHotCategorical
        q_s_dist = GumbelDistribution(temperature=self.gumbel_temp, logits=s_logits)
        log_qs_x = q_s_dist.log_prob(s_samples) # Use sampled s (soft or hard depending on rsample impl.)
        log_ps = self.prior_s.log_prob(s_samples)
        log_qs_x = torch.nan_to_num(log_qs_x, nan=-1e9, neginf=-1e9)
        log_ps = torch.nan_to_num(log_ps, nan=-1e9, neginf=-1e9)
        kl_s = log_qs_x - log_ps

        # KL[q(z|x,s) || p(z|s)]
        p_z_dist = self.prior_z(s_samples)
        kl_z = kl_divergence(q_z, p_z_dist).sum(dim=-1) # Sum over latent_dim_z

        kl_total = kl_s + kl_z # Shape (N,)

        # --- Reconstruction Term ---
        x_rec = self.decode(z=z_samples, s=s_samples)
        observed_mask = ~torch.isnan(x_original)

        if self.cfg.train.rec_loss == "mse":
            mse_per_element = F.mse_loss(x_rec, x_original, reduction="none")
            loss_rec_sum_observed = torch.sum(mse_per_element[observed_mask])
            num_observed = observed_mask.sum().float().clamp(min=1)
            loss_rec = loss_rec_sum_observed / num_observed
        elif self.cfg.train.rec_loss == "bce":
             x_original_01 = torch.nan_to_num(x_original, nan=0.5)
             x_rec_01 = torch.clamp(x_rec, 0.0, 1.0)
             bce_per_element = F.binary_cross_entropy(x_rec_01, x_original_01, reduction="none")
             loss_rec_sum_observed = torch.sum(bce_per_element[observed_mask])
             num_observed = observed_mask.sum().float().clamp(min=1)
             loss_rec = loss_rec_sum_observed / num_observed
        else:
            raise ValueError(f"Unknown rec_loss: {self.cfg.train.rec_loss}")

        # --- Total Loss ---
        kl_mean = kl_total.mean()
        # Use beta weights from config, ensuring they exist
        beta_s_val = self.cfg.hivae.get("beta_s", 1.0)
        beta_z_val = self.cfg.hivae.get("beta_z", 1.0)
        weighted_kl_mean = beta_s_val * kl_s.mean() + beta_z_val * kl_z.mean()
        total_loss = loss_rec + weighted_kl_mean

        return {
            "loss": total_loss, "rec": loss_rec,
            "kld": kl_mean, "kl_s": kl_s.mean(), "kl_z": kl_z.mean(),
        }


    def sample_z(self, num_samples: int, seed: int, tau: float = 1.0) -> Dict[str, torch.Tensor]:
        """Samples s from prior, then z from p(z|s). Returns dict for decode."""
        with seed_context(seed):
            s_sample_one_hot = self.prior_s.sample((num_samples,)).to(self.mydevice)
            p_z_dist = self.prior_z(s_sample_one_hot)
            std_scaled = p_z_dist.scale * tau
            z_sample = Normal(loc=p_z_dist.loc, scale=std_scaled).rsample()
        return {"z_sample": z_sample, "s_sample_one_hot": s_sample_one_hot}


    def forward(self, x: torch.Tensor, fill_evidence_on_reconstruction: bool = False) -> torch.Tensor:
        """ Performs reconstruction using the HI-VAE path (mode of s, mean of z). """
        x_original = x.to(self.mydevice) # Keep original with NaNs
        x_no_nan = torch.nan_to_num(x_original, nan=self.cfg.get("nan_mask_value", 0.0))

        # Encode x -> s logits
        if is_1d_data(self.cfg.dataset):
             s_logits = self.encoder_s(x_no_nan)
        else:
             s_logits = self.encoder_s(x_no_nan)

        # Get mode of s
        s_mode_indices = torch.argmax(s_logits, dim=-1)
        s_mode_one_hot = to_one_hot(s_mode_indices, self.L).float().to(self.mydevice)

        # Encode (x, s_mode) -> z params
        if is_1d_data(self.cfg.dataset):
             x_flat = x_no_nan.view(x.shape[0], -1)
             z_encoder_input = torch.cat((x_flat, s_mode_one_hot), dim=1)
             z_params = self.encoder_z(z_encoder_input)
        else: # 2D
             # Replicate the logic from encode() for 2D to get conv features then concat s
             conv_features = self.encoder_z._conv_1(x_no_nan)
             conv_features = F.relu(conv_features)
             conv_features = self.encoder_z._conv_2(conv_features)
             conv_features = F.relu(conv_features)
             for scale in self.encoder_z.scales:
                 conv_features = scale(conv_features)
                 conv_features = F.relu(conv_features)
             conv_features = self.encoder_z._conv_3(conv_features)
             conv_features = self.encoder_z._residual_stack(conv_features)
             if hasattr(self.encoder_z, '_conv_4'):
                  conv_features = self.encoder_z._conv_4(conv_features)
             conv_features_flat = conv_features.view(conv_features.size(0), -1)
             z_encoder_linear_input = torch.cat((conv_features_flat, s_mode_one_hot), dim=1)
             z_params = self.encoder_z.linear(z_encoder_linear_input)


        z_mean_q, _ = z_params.chunk(2, dim=1)

        # Decode (z_mean, s_mode) -> x_rec
        x_reconstructed = self.decode(z=z_mean_q, s=s_mode_one_hot)

        if fill_evidence_on_reconstruction:
            original_observed_mask = ~torch.isnan(x_original)
            x_reconstructed = torch.where(original_observed_mask, x_original, x_reconstructed)

        # Ensure output shape matches input shape
        return x_reconstructed.view_as(x)


    def impute(self, x_original_flat_with_nans: torch.Tensor) -> torch.Tensor:
        """ Imputes missing values (NaNs) using the forward reconstruction. """
        self.eval()
        with torch.no_grad():
            imputed_flat = self.forward(x_original_flat_with_nans, fill_evidence_on_reconstruction=False)
        return imputed_flat
