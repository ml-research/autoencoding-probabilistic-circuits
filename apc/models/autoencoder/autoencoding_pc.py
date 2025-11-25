from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from apc.models.abstract_model import AbstractPcAutoencoder
from apc.models.decoder.nn_decoder import NeuralDecoder1D, NeuralDecoder2D
from apc.models.encoder.abstract_encoder import AbstractEncoder
from apc.models.encoder.conv_pc_encoder import ConvPcEncoder
from apc.models.encoder.conv_pc_spnae_encoder import ConvPcSpnAeEncoder
from apc.models.encoder.einet_encoder import EinetEncoder
from apc.models.encoder.einet_spnae_encoder import EinetSpnaeEncoder
from apc.enums import PcEncoderType, DecoderType
from simple_einet.data import is_1d_data, is_debd_data


class EmptyModule(nn.Module):
    def __init__(self):
        super().__init__()


class APC(AbstractPcAutoencoder):

    def decoder_types(self) -> list[DecoderType]:
        if self.cfg.apc.decoder == DecoderType.PC:
            return [DecoderType.PC]
        else:
            return [DecoderType.NN, DecoderType.PC]

    def _default_decoder_type(self) -> DecoderType:
        return DecoderType.NN

    def make_encoder(self) -> AbstractEncoder:
        match self.cfg.apc.encoder:

            case PcEncoderType.EINET:
                return EinetEncoder(self.cfg)

            case PcEncoderType.EINET_ACT | PcEncoderType.EINET_CAT:
                assert np.isclose(self.cfg.weight.kld, 0.0), "EinetSpnaeEncoder does not support KL divergence."
                return EinetSpnaeEncoder(self.cfg)

            case PcEncoderType.CONV_PC:
                if is_1d_data(self.cfg.dataset):
                    assert (
                        self.cfg.conv_pc.kernel_size_h == 1 or self.cfg.conv_pc.kernel_size_w == 1
                    ), "conv_pc.kernel_size_h or conv_pc.kernel_size_w must be 1 for 1D data."
                return ConvPcEncoder(self.cfg)

            case PcEncoderType.CONV_PC_SPNAE_ACT | PcEncoderType.CONV_PC_SPNAE_CAT:
                assert np.isclose(self.cfg.weight.kld, 0.0), "ConvPcSpnAeEncoder does not support KL divergence."
                assert not is_1d_data(
                    self.cfg.dataset
                ), f"ConvPcActEncoder is only supported for 2D data but {self.cfg.dataset} is 1D."
                return ConvPcSpnAeEncoder(self.cfg)

            case _:
                raise ValueError(f"Unknown encoder {self.cfg.apc.encoder}.")

    def make_decoder(self) -> EmptyModule | NeuralDecoder1D | NeuralDecoder2D:
        if self.cfg.apc.decoder == DecoderType.PC:
            return EmptyModule()

        if is_1d_data(self.cfg.dataset):
            return NeuralDecoder1D(cfg=self.cfg)
        else:
            return NeuralDecoder2D(cfg=self.cfg)

    def reconstruct(self, x: torch.Tensor, decoder_type: DecoderType = None, fill_evidence=False) -> torch.Tensor:
        mask = x.isnan()
        z = self.encode(x)

        x_rec = self.decode(z, mpe=True, decoder_type=decoder_type, fill_evidence=fill_evidence)
        if mask.any() and fill_evidence:
            x_rec[~mask] = x[~mask].to(dtype=x_rec.dtype)


        # if is_debd_data(self.cfg.dataset):
        #     # x_rec are probabilities in [0, 1], make them into binary values
        #     x_rec = (x_rec > 0.5).float()

        return x_rec.view(x_rec.shape[0], *self.data_shape)

    def encode(
        self,
        x: torch.Tensor,
        mpe=True,
        tau: float = 1.0,
        return_leaf_params=False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Encode x
        z = self.encoder.encode(x, mpe=mpe, return_leaf_params=return_leaf_params, tau=tau)

        if return_leaf_params:
            z_mu, z_logvar, z_samples = z
        else:
            z_samples = z

        if return_leaf_params:
            return z_mu, z_logvar, z_samples
        else:
            return z_samples

    def decode(
        self,
        z: torch.Tensor,
        mpe=True,
        tau: float = 1.0,
        decoder_type: Optional[DecoderType] = None,
        fill_evidence=False,
    ) -> torch.Tensor:
        # Sample x' ~ p( x | z )
        # Construct evidence

        if decoder_type is None:
            decoder_type = self.cfg.apc.decoder

        if decoder_type == DecoderType.NN:
            # Forward pass through NN decoder
            x_rec = self.decoder(z).view(-1, *self.data_shape)  # Output is Linear(...)
            if not self._is_1d_data:  # Scale if we are modeling images
                x_rec = (x_rec + 1) / 2 * (2**self.cfg.n_bits - 1)


            return x_rec

        elif decoder_type == DecoderType.PC:

            # Decode using PC encoder
            x_rec = self.encoder.decode(z=z, mpe=mpe, tau=tau, fill_evidence=fill_evidence)
            return x_rec
        else:
            raise ValueError(f"Unknown decoder {decoder_type}.")

    def sample_x(self, num_samples: int, seed: int, decoder_type: DecoderType = None, tau: float = 1.0) -> torch.Tensor:
        if decoder_type == DecoderType.PC:
            x = self.encoder.sample(
                num_samples=num_samples,
                seed=seed,
                tau=tau,
                return_leaf_params=False,
                return_latents=False,
                return_data=True,
                mpe=False,
            )
        elif decoder_type == DecoderType.NN:
            x = super().sample_x(num_samples, seed, decoder_type=decoder_type, tau=tau)

        return x

    def sample_z(
        self,
        num_samples: int,
        seed: int,
        return_leaf_params: bool = False,
        tau: float = 1.0,
    ) -> torch.Tensor:
        z = self.encoder.sample_z(
            num_samples=num_samples,
            seed=seed,
            return_leaf_params=return_leaf_params,
            tau=tau,
        )
        return z

    def log_likelihood(self, x: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder.log_likelihood(x, z)

    def loss(self, data) -> dict[str, torch.Tensor]:
        # Batch-size
        batch_size = data.shape[0]

        loss_dict = {}

        # Encode data with PC
        if self.cfg.weight.kld > 0.0 or self.cfg.weight.rec > 0.0 or self.cfg.nll_x_and_z:
            z_mu, z_logvar, z_samples = self.encode(data, mpe=False, return_leaf_params=True)

        ###########################
        # Negative Log-likelihood #
        ###########################
        if self.cfg.weight.nll > 0.0:
            lls = self.log_likelihood(x=data, z=z_samples if self.cfg.nll_x_and_z else None)
            loss_dict["nll"] = -1 * lls.sum() / batch_size

        ##################
        # Reconstruction #
        ##################
        if self.cfg.weight.rec > 0.0:
            if self.cfg.apc.train_decode_mpe:
                z_to_decode = z_mu
            else:
                z_to_decode = z_samples
            x_rec = self.decode(z_to_decode, decoder_type=self.cfg.apc.decoder)

            if not self._is_1d_data:  # Image data, scale to [-1, 1]
                # NOTE: This is wrong and should be x / (2**n_bits - 1) * 2 - 1 instead. In principle, this will simply "wronlgy" scale x_rec and data, which is then used to compute the MSE loss.
                # This is not a problem, as long as the scaling is consistent. The MSE test evaluations use the correct scaling.
                # Therefore, to keep results consistent, we keep this "wrong" scaling for now.
                x_rec = x_rec / (self.cfg.n_bits**2 - 1) * 2 - 1
                data = data / (self.cfg.n_bits**2 - 1) * 2 - 1
                # x_rec = x_rec / (2 ** self.cfg.n_bits - 1) * 2 - 1
                # data = data / (2 ** self.cfg.n_bits - 1) * 2 - 1


            if self.cfg.train.rec_loss == "mse":
                loss_dict["rec"] = F.mse_loss(x_rec, data, reduction="sum") / batch_size
            elif self.cfg.train.rec_loss == "bce":
                # Scale from [-1, 1] to [0, 1]
                # x_rec = (x_rec + 1) / 2
                # data = (data + 1) / 2
                loss_dict["rec"] = F.binary_cross_entropy(x_rec, data, reduction="sum") / batch_size
            else:
                raise ValueError(f"Unknown rec_loss: {self.cfg.train.rec_loss}")

        #################
        # KL-Divergence #
        #################
        if self.cfg.weight.kld > 0.0:
            # Compute kl divergence to N(0, I)
            loss_dict["kld"] = -0.5 * torch.sum(1 + z_logvar - z_mu**2 - z_logvar.exp()) / batch_size
        return loss_dict


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config")
    def main_hydra(cfg: DictConfig):
        size = 32
        bs = 64
        # z = torch.randn(size=(1, 1, cfg.latent_dim))
        x = torch.randint(low=0, high=255, size=(bs, 1, size, size)).float()
        apc = APC(cfg)
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    main_hydra()
