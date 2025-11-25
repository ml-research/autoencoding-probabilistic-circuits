import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple
from apc.models.autoencoder.vanilla_autoencoder import VanillaAutoencoder

# --- Assume previous definitions are available ---
# ParallelLinear class
# ParallelVAE class
# DependencyNetworkVAE class (potentially inheriting from VanillaAutoencoder)
# Placeholder for VanillaAutoencoder if needed


class ParallelLinear(nn.Module):
    """Applies D independent linear transformations to the input data.

    Args:
        num_parallel_dims (int): D, the number of parallel transformations.
        in_features (int): Size of each input sample feature per parallel dim.
        out_features (int): Size of each output sample feature per parallel dim.
        bias (bool): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """

    def __init__(self, num_parallel_dims: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.num_parallel_dims = num_parallel_dims
        self.in_features = in_features
        self.out_features = out_features

        # Weight: (D, out_features, in_features)
        self.weight = nn.Parameter(torch.empty((num_parallel_dims, out_features, in_features)))
        if bias:
            # Bias: (D, out_features)
            self.bias = nn.Parameter(torch.empty(num_parallel_dims, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights and biases similar to nn.Linear
        for i in range(self.num_parallel_dims):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, D, in_features)
                               or (batch_size, D * in_features) - will be reshaped.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, D, out_features)
        """
        batch_size = x.shape[0]

        # Reshape input if it's flattened (batch_size, D * in_features)
        if x.dim() == 2 and x.shape[1] == self.num_parallel_dims * self.in_features:
            x = x.view(batch_size, self.num_parallel_dims, self.in_features)
        elif x.dim() != 3 or x.shape[1] != self.num_parallel_dims or x.shape[2] != self.in_features:
            raise ValueError(
                f"Input tensor shape {x.shape} is incompatible. "
                f"Expected (batch, D, in_features) or (batch, D*in_features). "
                f"D={self.num_parallel_dims}, in_features={self.in_features}"
            )

        output = torch.einsum("bdi,doi->bdo", x, self.weight)

        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)  # Add broadcasted bias

        return output

    def extra_repr(self) -> str:
        return "num_parallel_dims={}, in_features={}, out_features={}, bias={}".format(
            self.num_parallel_dims, self.in_features, self.out_features, self.bias is not None
        )


class ParallelMarginalVAE(nn.Module):
    """Parallelized Marginal VAE for Stage 1 of VAEM."""

    def __init__(self, cfg, num_dimensions: int, feature_dim: int, latent_dim: int, data_type: str):
        super().__init__()
        self.cfg = cfg
        self.num_dimensions = num_dimensions  # D
        self.feature_dim = feature_dim  # Input dim per parallel dim
        self.latent_dim = latent_dim  # Latent dim per parallel dim
        self.data_type = data_type  # Assumed same for all D

        hidden_dim = cfg.model.marginal_hidden_dim

        self.encoder_net = nn.Sequential(
            ParallelLinear(num_dimensions, feature_dim, hidden_dim),
            nn.ReLU(),
            ParallelLinear(num_dimensions, hidden_dim, 2 * latent_dim),
        )

        output_dim = self._get_output_dim_per_dim()
        self.decoder_net = nn.Sequential(
            ParallelLinear(num_dimensions, latent_dim, hidden_dim),
            nn.ReLU(),
            ParallelLinear(num_dimensions, hidden_dim, output_dim),
        )

    def _get_output_dim_per_dim(self) -> int:
        if self.data_type == "continuous":
            return self.feature_dim
        elif self.data_type == "binary":
            return self.feature_dim
        elif self.data_type == "categorical":
            if not hasattr(self.cfg.model, "num_categories") or not isinstance(self.cfg.model.num_categories, int):
                raise ValueError("cfg.model.num_categories (int) required for categorical data type in ParallelVAE")
            return self.cfg.model.num_categories
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def __reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        if x.dim() == 2 and x.shape[1] == self.num_dimensions * self.feature_dim:
            x = x.view(batch_size, self.num_dimensions, self.feature_dim)
        elif x.dim() != 3 or x.shape[1] != self.num_dimensions or x.shape[2] != self.feature_dim:
            raise ValueError(f"Input tensor shape {x.shape} is incompatible.")

        mu_logvar = self.encoder_net(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

    def encode(self, x: torch.Tensor, mpe=True) -> torch.Tensor:
        mu, logvar = self.encode_params(x)
        if mpe:
            z = mu
        else:
            z = self.__reparameterize(mu=mu, logvar=logvar)
        return z  # Shape: (batch_size, D, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (batch_size, D, latent_dim)
        return self.decoder_net(z)  # Shape: (batch, D, output_dim_per_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode_params(x)
        z = self.__reparameterize(mu, logvar)
        recon_params = self.decode(z)
        return {"recon_params": recon_params, "mu": mu, "logvar": logvar, "z": z}

    def loss_function(self, forward_outputs: Dict[str, torch.Tensor], x: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon_params = forward_outputs["recon_params"]
        mu = forward_outputs["mu"]
        logvar = forward_outputs["logvar"]
        batch_size = mu.shape[0]

        if x.dim() == 2 and x.shape[1] == self.num_dimensions * self.feature_dim:
            x = x.view(batch_size, self.num_dimensions, self.feature_dim)
        elif x.dim() != 3 or x.shape[1] != self.num_dimensions or x.shape[2] != self.feature_dim:
            raise ValueError(f"Input tensor x shape {x.shape} incompatible for loss.")

        if self.data_type == "continuous":
            rec_loss = F.mse_loss(recon_params, x, reduction="none").sum(dim=[1, 2])
        elif self.data_type == "binary":
            rec_loss = F.binary_cross_entropy_with_logits(recon_params, x, reduction="none").sum(dim=[1, 2])
        elif self.data_type == "categorical":
            num_categories = recon_params.shape[-1]
            recon_params_flat = recon_params.reshape(-1, num_categories)
            x_flat = x.reshape(-1).long()
            rec_loss = (
                F.cross_entropy(recon_params_flat, x_flat, reduction="none")
                .view(batch_size, self.num_dimensions)
                .sum(dim=1)
            )
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
        rec_loss = rec_loss.sum()

        kld_loss_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss_per_element.sum(dim=[1, 2]).sum()

        return {"rec": rec_loss / batch_size, "kld": kld_loss / batch_size}


class DependencyNetworkVAE(VanillaAutoencoder):
    def __init__(self, cfg, input_latent_dim_sum: int, dependency_latent_dim: int):
        super().__init__(cfg, data_shape=(input_latent_dim_sum,))  # Input is flattened z
        self.input_latent_dim_sum = input_latent_dim_sum
        self.dependency_latent_dim = dependency_latent_dim  # Latent dim 'h'

        hidden_dim = self.cfg.model.dependency_hidden_dim
        # Encoder: flattened z -> h params
        self.encoder = nn.Sequential(
            nn.Linear(input_latent_dim_sum, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * dependency_latent_dim),  # mu_h, logvar_h
        )
        # Decoder: h -> z_recon params (mu_z_recon, logvar_z_recon)
        self.decoder = nn.Sequential(
            nn.Linear(dependency_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * input_latent_dim_sum),  # mu_z, logvar_z
        )

    def __reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, z_concat_flat: torch.Tensor, mpe=True) -> torch.Tensor:
        # z_concat_flat shape: (batch_size, input_latent_dim_sum)
        mu_h, logvar_h = self.encoder(z_concat_flat).chunk(2, dim=1)
        if mpe:
            h = mu_h
        else:
            h = self.__reparameterize(mu=mu_h, logvar=logvar_h)
        return h  # Shape: (batch_size, dependency_latent_dim)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        # h shape: (batch_size, dependency_latent_dim)
        # Output represents parameters for p(z|h) = N(mu_z, exp(logvar_z))
        recon_z_params = self.decoder(h)  # Shape: (batch_size, 2 * input_latent_dim_sum)
        return recon_z_params

    def forward(self, z_concat_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu_h, logvar_h = self.encoder(z_concat_flat).chunk(2, dim=1)
        h = self.__reparameterize(mu_h, logvar_h)
        recon_z_params = self.decode(h)
        return {"recon_z_params": recon_z_params, "mu_h": mu_h, "logvar_h": logvar_h, "h": h}

    def loss(self, z_concat_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        forward_outputs = self.forward(z_concat_flat)
        recon_z_params = forward_outputs["recon_z_params"]
        mu_h = forward_outputs["mu_h"]
        logvar_h = forward_outputs["logvar_h"]
        batch_size = z_concat_flat.shape[0]

        mu_z_recon, logvar_z_recon = recon_z_params.chunk(2, dim=1)

        # Reconstruction Loss: log p(z|h) - Assuming factorized Gaussian
        recon_loss_z = 0.5 * torch.sum(logvar_z_recon + (z_concat_flat - mu_z_recon).pow(2) / logvar_z_recon.exp())

        # KLD Loss: KL(q(h|z) || p(h)) - Assuming p(h) = N(0,I)
        kld_loss_h = -0.5 * torch.sum(1 + logvar_h - mu_h.pow(2) - logvar_h.exp())

        return {"rec_z": recon_loss_z / batch_size, "kld_h": kld_loss_h / batch_size}


# --- Full VAEM Model using ParallelVAE ---
class VAEM(nn.Module):
    def __init__(self, cfg, parallel_marginal_vae: ParallelMarginalVAE, dependency_vae: DependencyNetworkVAE):
        """Initializes the VAEM model using pre-trained components.

        Args:
            cfg: Configuration object.
            parallel_marginal_vae (ParallelVAE): The trained Stage 1 VAE.
            dependency_vae (DependencyNetworkVAE): The trained Stage 2 VAE.
        """
        super().__init__()
        self.cfg = cfg
        # Store trained modules
        self.parallel_marginal_vae = parallel_marginal_vae
        self.dependency_vae = dependency_vae

        # Infer dimensions from components
        self.num_dimensions = parallel_marginal_vae.num_dimensions  # D
        self.feature_dim = parallel_marginal_vae.feature_dim
        self.marginal_latent_dim = parallel_marginal_vae.latent_dim
        self.dependency_latent_dim = dependency_vae.dependency_latent_dim
        self.data_type = parallel_marginal_vae.data_type  # Store assumed data type

        # Freeze parameters of components trained in stages
        for p in self.parallel_marginal_vae.parameters():
            p.requires_grad = False
        for p in self.dependency_vae.parameters():
            p.requires_grad = False

        self.parallel_marginal_vae.eval()
        self.dependency_vae.eval()

    def encode(self, x_batch: torch.Tensor, mpe=True) -> torch.Tensor:
        """Encodes the input batch x -> z -> h.
        x_batch shape: (batch_size, D * feature_dim) or (batch_size, D, feature_dim)
        """
        # Stage 1: Encode x -> z
        # z shape: (batch_size, D, marginal_latent_dim)
        z = self.parallel_marginal_vae.encode(x_batch, mpe=mpe)

        # Prepare z for Stage 2 encoder: flatten
        # z_flat shape: (batch_size, D * marginal_latent_dim)
        z_flat = z.view(z.shape[0], -1)

        # Stage 2: Encode z -> h
        # h shape: (batch_size, dependency_latent_dim)
        h = self.dependency_vae.encode(z_flat, mpe=mpe)
        return h

    def decode(self, h: torch.Tensor, sample_x=False) -> torch.Tensor:
        """Decodes h -> z -> x.
        h shape: (batch_size, dependency_latent_dim)
        Returns: Tensor of reconstruction parameters or samples.
                 Shape: (batch_size, D, output_dim_per_dim)
        """
        # Stage 2: Decode h -> z_recon parameters
        # recon_z_params shape: (batch_size, 2 * D * marginal_latent_dim)
        recon_z_params = self.dependency_vae.decode(h)
        mu_z_recon, logvar_z_recon = recon_z_params.chunk(2, dim=1)

        # Sample z (or use mean mu_z_recon for reconstruction)
        # Need to reshape mu/logvar before reparameterization to match ParallelVAE input
        batch_size = h.shape[0]
        mu_z_reshaped = mu_z_recon.view(batch_size, self.num_dimensions, self.marginal_latent_dim)
        logvar_z_reshaped = logvar_z_recon.view(batch_size, self.num_dimensions, self.marginal_latent_dim)
        z_recon = self.parallel_marginal_vae._ParallelVAE__reparameterize(mu_z_reshaped, logvar_z_reshaped)
        # z_recon shape: (batch_size, D, marginal_latent_dim)

        # Stage 1: Decode z -> x parameters
        # x_recon_params shape: (batch_size, D, output_dim_per_dim)
        x_recon_params = self.parallel_marginal_vae.decode(z_recon)

        if sample_x:
            # Sample final x values from the output distributions
            # Adapting sampling logic from ParallelVAE loss
            if self.data_type == "continuous":
                # Assume fixed variance or learn it
                std_dev = self.cfg.model.get("output_std_dev", 0.1)  # Example
                x_samples = torch.normal(x_recon_params, std=std_dev)
            elif self.data_type == "binary":
                probs = torch.sigmoid(x_recon_params)
                x_samples = torch.bernoulli(probs)
            elif self.data_type == "categorical":
                num_categories = x_recon_params.shape[-1]
                probs = F.softmax(x_recon_params.view(-1, num_categories), dim=-1)
                x_indices = torch.multinomial(probs, num_samples=1)
                x_samples = x_indices.view(batch_size, self.num_dimensions, 1)  # Keep shape consistent
            else:
                x_samples = x_recon_params  # Fallback
            return x_samples  # Shape: (batch, D, feature_dim or 1)
        else:
            # Return the parameters themselves (e.g., mean for continuous, logits for discrete)
            return x_recon_params  # Shape: (batch, D, output_dim_per_dim)

    def generate(self, num_samples: int = 1, sample_x: bool = True) -> torch.Tensor:
        """Generates new samples: sample h ~ p(h), then decode h -> z -> x"""
        # Sample h from prior (e.g., standard normal)
        h_sample = torch.randn(
            num_samples, self.dependency_latent_dim, device=self.dependency_vae.encoder[0].weight.device
        )  # Get device from a layer
        # Decode h -> z -> x
        x_generated = self.decode(h_sample, sample_x=sample_x)
        return x_generated

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Convenience method for reconstruction: encode then decode"""
        h = self.encode(x_batch, mpe=True)  # Use mpe=True for deterministic reconstruction
        x_recon_params = self.decode(h, sample_x=False)  # Get reconstruction parameters
        return x_recon_params


# --- Updated Training Logic Outline ---

# config = load_your_config()
# dataset = load_your_homogeneous_dataset() # Assumes ParallelVAE used on homogeneous data
# D = determine_num_dimensions(dataset)
# feature_dim = determine_feature_dim(dataset)
# data_type = determine_data_type(config) # Single data type

# # Stage 1: Train ParallelVAE
# print("Training Stage 1: ParallelVAE")
# parallel_marginal_vae = ParallelVAE(config, D, feature_dim, config.model.marginal_latent_dim, data_type)
# optimizer_stage1 = torch.optim.Adam(parallel_marginal_vae.parameters(), lr=config.train.lr_stage1)
# dataloader = create_dataloader(dataset, config.train.batch_size) # Yields (batch, D*feat) or (batch, D, feat)

# for epoch in range(config.train.epochs_stage1):
#     parallel_marginal_vae.train()
#     for batch_x in dataloader:
#         optimizer_stage1.zero_grad()
#         forward_outputs = parallel_marginal_vae(batch_x)
#         loss_dict = parallel_marginal_vae.loss_function(forward_outputs, batch_x)
#         total_loss = loss_dict["rec"] + config.train.beta * loss_dict["kld"]
#         total_loss.backward()
#         optimizer_stage1.step()
#     # Add logging/validation
# parallel_marginal_vae.eval()


# # Stage 2: Prepare data (encode full dataset with ParallelVAE)
# print("Preparing data for Stage 2...")
# all_z_flat = []
# full_dataloader = create_dataloader(dataset, config.train.batch_size, shuffle=False)
# with torch.no_grad():
#     for batch_x in full_dataloader:
#         # z shape: (batch, D, marginal_latent_dim)
#         z = parallel_marginal_vae.encode(batch_x, mpe=False) # Sample z for training stage 2
#         # Flatten z for DependencyNetworkVAE input
#         z_flat = z.view(z.shape[0], -1) # (batch, D * marginal_latent_dim)
#         all_z_flat.append(z_flat)
# z_dataset_tensor = torch.cat(all_z_flat, dim=0)
# z_dataloader = create_dataloader(TensorDataset(z_dataset_tensor), config.train.batch_size, shuffle=True)

# # Stage 2: Train Dependency Network
# print("Training Stage 2: Dependency Network")
# input_latent_dim_sum = D * config.model.marginal_latent_dim
# dependency_latent_dim = config.model.dependency_latent_dim
# dependency_vae = DependencyNetworkVAE(config, input_latent_dim_sum, dependency_latent_dim)
# optimizer_stage2 = torch.optim.Adam(dependency_vae.parameters(), lr=config.train.lr_stage2)

# for epoch in range(config.train.epochs_stage2):
#     dependency_vae.train()
#     for batch_z_flat, in z_dataloader: # Input is flattened z
#         optimizer_stage2.zero_grad()
#         loss_dict = dependency_vae.loss(batch_z_flat) # loss takes flattened z
#         total_loss = loss_dict["rec_z"] + config.train.gamma * loss_dict["kld_h"]
#         total_loss.backward()
#         optimizer_stage2.step()
#     # Add logging/validation
# dependency_vae.eval()

# # --- Final VAEM Model ---
# print("Creating final VAEM model...")
# vaem_model = VAEM(config, parallel_marginal_vae, dependency_vae)

# # Use vaem_model for encoding, generation, etc.
# x_recon = vaem_model(some_input_batch)
# generated_x = vaem_model.generate(num_samples=10)
