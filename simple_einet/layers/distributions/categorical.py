import torch
from typing import Tuple
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from simple_einet.sampling_utils import SamplingContext, sample_categorical_differentiably


class Categorical(AbstractLeaf):
    """Categorical layer. Maps each input feature to its categorical log likelihood.

    Probabilities are modeled as unconstrained parameters and are transformed via a softmax function into [0, 1] when needed.
    """

    def __init__(self, num_features: int, num_channels: int, num_leaves: int, num_repetitions: int, num_bins: int):
        """
        Initializes a categorical distribution with the given parameters.

        Args:
            num_features (int): The number of features in the input data.
            num_channels (int): The number of channels in the input data.
            num_leaves (int): The number of leaves in the tree structure.
            num_repetitions (int): The number of repetitions for each leaf.
            num_bins (int): The number of bins for the categorical distribution.
        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create logits
        self.logits = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions, num_bins))

    def _get_base_distribution(self, ctx: SamplingContext = None):
        # Use sigmoid to ensure, that probs are in valid range
        if ctx is not None and ctx.is_differentiable:
            return CustomCategorical(logits=self.logits)
        else:
            return dist.Categorical(logits=self.logits)


    def get_params(self):
        return self.logits


class CustomCategorical:
    """
    A custom implementation of the Categorical distribution.

    Sampling from this distribution is differentiable.

    Args:
        logits (torch.Tensor): The logits of the Categorical distribution.
    """

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def sample(self, sample_shape: Tuple[int]) -> torch.Tensor:
        """
        Generates random samples from the categorical distribution.

        Args:
            sample_shape (Tuple[int]): The shape of the desired output tensor.

        Returns:
            samples (torch.Tensor): A tensor of shape `sample_shape` containing random samples from the Categorical distribution.
        """
        num_samples = sample_shape[0]
        logits = self.logits.unsqueeze(0)
        logits = self.logits.expand(num_samples, -1, -1, -1, -1, -1, -1)
        samples = sample_categorical_differentiably(logits=logits, is_mpe=False, hard=False, tau=1.0, dim=-1)

        # Convert one-hot samples to indices
        samples = (samples * torch.arange(0, self.logits.shape[-1], device=samples.device)).sum(dim=-1)
        return samples

    def mpe(self, num_samples) -> torch.Tensor:
        """
        Generates MPE samples from the Categorical distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            samples (torch.Tensor): A tensor of shape `num_samples + logits.shape` containing MPE samples from the Categorical distribution.
        """
        mpe = sample_categorical_differentiably(logits=self.logits, is_mpe=True, hard=False, tau=1.0, dim=-1)

        # Convert one-hot mpe to indices
        mpe = (mpe * torch.arange(0, self.logits.shape[-1], device=mpe.device)).sum(dim=-1)

        return mpe.repeat(num_samples, 1, 1, 1, 1)

    def log_prob(self, x):
        """
        Computes the log probability density of the Categorical distribution at the given value.

        Args:
            x (torch.Tensor): The value(s) at which to evaluate the log probability density.

        Returns:
            torch.Tensor: The log probability density of the Categorical distribution at the given value(s).
        """
        return dist.Categorical(logits=self.logits).log_prob(x)

    def get_params(self):
        return self.logits.unsqueeze(-1)
