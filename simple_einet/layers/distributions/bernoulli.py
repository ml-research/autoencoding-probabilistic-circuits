from typing import Tuple

import torch
from torch import distributions as dist
from torch import nn

from simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from simple_einet.sampling_utils import SamplingContext, SIMPLE


class Bernoulli(AbstractLeaf):
    """Bernoulli layer. Maps each input feature to its bernoulli log likelihood.

    Probabilities are modeled as unconstrained parameters and are transformed via a sigmoid function into [0, 1] when needed.
    """

    def __init__(self, num_features: int, num_channels: int, num_leaves: int, num_repetitions: int):
        """
        Initializes a Bernoulli distribution with the given parameters.

        Args:
            num_features (int): The number of features in the input data.
            num_channels (int): The number of channels in the input data.
            num_leaves (int): The number of leaves in the tree structure.
            num_repetitions (int): The number of repetitions for each leaf.
        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create bernoulli parameters
        self.logits = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions))

    def _get_base_distribution(self, ctx: SamplingContext = None):
        # Use sigmoid to ensure, that probs are in valid range
        probs = torch.sigmoid(self.logits)
        if ctx is not None and ctx.is_differentiable:
            return DifferentiableBernoulli(probs=probs)
        else:
            return dist.Bernoulli(probs=probs)

    def get_params(self):
        return self.logits.unsqueeze(-1)

class DifferentiableBernoulli:
    def __init__(self, probs):
        self.probs = probs

    def sample(self, sample_shape: Tuple[int], tau=1.0, mpe=False):
        probs = self.probs
        probs = (probs ** (1 / tau)) / ((probs ** (1 / tau)) + ((1 - probs) ** (1 / tau)))
        probs = probs.expand(sample_shape + probs.shape)

        probs = probs.clamp(min=1e-7, max=1 - 1e-7)

        log_weights = torch.stack([probs.log(), (1 - probs).log()], dim=0)
        samples_oh = SIMPLE(log_weights=log_weights, dim=0, is_mpe=mpe)
        rnge = torch.arange(2, device=probs.device).view((-1,) + (1,) * (len(samples_oh.shape) - 1))
        samples = torch.sum(samples_oh * rnge, dim=0)
        return samples.float()

    def log_prob(self, x):
        return dist.Bernoulli(probs=self.probs).log_prob(x.float())

    def get_params(self):
        return self.probs.unsqueeze(-1)




if __name__ == '__main__':
    # Test Bernoulli
    num_features = 10
    num_channels = 3
    num_leaves = 4
    num_repetitions = 1

    bernoulli = Bernoulli(num_features, num_channels, num_leaves, num_repetitions)

    # Test forward pass
    x = torch.randint(0, 2, size=(5, num_channels, num_features)).float()
    # log_likelihood = bernoulli(x, marginalized_scopes=None)
    # assert log_likelihood.shape == (5, num_channels, num_features, num_leaves, num_repetitions)
    # print('Bernoulli test successful!')
    #
    # # LLS: Check if gradients work w.r.t. the parameters
    # bernoulli.zero_grad()
    # log_likelihood = bernoulli(x, marginalized_scopes=None)
    # log_likelihood.sum().backward()
    # print(bernoulli.logits.grad.sum())


    # Sampling: Check if gradients work w.r.t. the parameters
    # bernoulli.zero_grad()
    ctx = SamplingContext(num_samples=5, indices_repetition=torch.ones(1).requires_grad_(True), is_differentiable=True)
    samples = bernoulli.sample(ctx=ctx)
    print(samples)
    # (samples.mean(3) - x).abs().sum().backward()
    # print(bernoulli.logits.grad)
