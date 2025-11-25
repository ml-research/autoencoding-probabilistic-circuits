"""
Source: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
"""

import torch
from torch import nn, Tensor, quantized_gru_cell
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, decay: float = 0.99, epsilon: float = 1e-5, ema = True):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon
        self.ema = ema


        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        # Initialize EMA variables
        if ema:
            self.register_buffer('ema_cluster_size', torch.zeros(self.K))
            self.ema_w = nn.Parameter(self.embedding.weight.clone())
            self.ema_w = nn.Parameter(torch.Tensor(self.K, self.D))
            self.ema_w.data.normal_()


    def forward(self, latents: Tensor, return_loss=False) -> tuple[Tensor, Tensor]:
        # latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        if self.ema and self.training:
            # EMA updates
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * torch.sum(encoding_one_hot, dim=0)

            # Add small value to avoid dividing by zero
            n = torch.sum(self.ema_cluster_size) + self.epsilon
            self.ema_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.K * self.epsilon) * n

            dw = torch.matmul(flat_latents.t(), encoding_one_hot)
            self.ema_w = nn.Parameter(self.decay * self.ema_w + (1 - self.decay) * dw.t())
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))



        if return_loss:
            # Compute the VQ Losses
            commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
            embedding_loss = F.mse_loss(quantized_latents, latents.detach())

            vq_loss = commitment_loss * self.beta + embedding_loss
        else:
            vq_loss = None

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # Permute back
        # quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous() # [B x D x H x W]

        return quantized_latents, vq_loss


if __name__ == "__main__":
    vq = VectorQuantizer(num_embeddings=10, embedding_dim=32)
    z = torch.randn(4, 32, 8, 8)
    z_q, loss = vq(z)
    print(z_q.shape, loss)
