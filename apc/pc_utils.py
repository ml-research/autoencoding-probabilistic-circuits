from omegaconf import DictConfig

from simple_einet.data import is_1d_data
from simple_einet.layers.distributions.bernoulli import Bernoulli
from simple_einet.layers.distributions.binomial import Binomial
from simple_einet.layers.distributions.categorical import Categorical
from simple_einet.layers.distributions.normal import Normal, RatNormal


def get_leaf_args(cfg: DictConfig) -> tuple:
    """
    Get the leaf arguments for the Einet model.

    Args:
        cfg: The configuration object.

    Returns:
        tuple: The leaf arguments and the leaf type.
    """
    if cfg.einet.dist == "binomial":
        leaf_type = Binomial
        if is_1d_data(cfg.dataset):
            total_count = 1
        else:
            total_count = 2**cfg.n_bits - 1
        leaf_kwargs = {"total_count": total_count}
    elif cfg.einet.dist == "bernoulli":
        leaf_type = Bernoulli
        leaf_kwargs = {}
    elif cfg.einet.dist == "normal":
        leaf_type = Normal
        leaf_kwargs = {}
    elif cfg.einet.dist == "normal_rat":
        leaf_type = RatNormal
        leaf_kwargs = {
            "min_sigma": cfg.normal_rat.min_std,
            "max_sigma": cfg.normal_rat.max_std,
            "min_mean": cfg.normal_rat.min_mean,
            "max_mean": cfg.normal_rat.max_mean,
        }
    elif cfg.einet.dist == "categorical":
        leaf_type = Categorical
        if is_1d_data(cfg.dataset):
            num_bins = 2
        else:
            num_bins = 2**cfg.n_bits - 1
        leaf_kwargs = {"num_bins": num_bins}
    else:
        raise ValueError(f"Unknown distribution {cfg.einet.dist}.")
    return leaf_kwargs, leaf_type



def get_latent_leaf_args(cfg: DictConfig) -> tuple:
    """
    Get the leaf arguments for the Einet model.

    Args:
        cfg: The configuration object.

    Returns:
        tuple: The leaf arguments and the leaf type.
    """
    if cfg.latent_dist.type == "binomial":
        leaf_type = Binomial
        leaf_kwargs = {"total_count": cfg.latent_dist.binomial.N}
    elif cfg.latent_dist.type == "bernoulli":
        leaf_type = Bernoulli
        leaf_kwargs = {}
    elif cfg.latent_dist.type == "normal":
        leaf_type = Normal
        leaf_kwargs = {}
    elif cfg.latent_dist.type == "normal_rat":
        leaf_type = RatNormal
        leaf_kwargs = {
            "min_sigma": cfg.latent_dist.normal_rat.min_std,
            "max_sigma": cfg.latent_dist.normal_rat.max_std,
            "min_mean": cfg.latent_dist.normal_rat.min_mean,
            "max_mean": cfg.latent_dist.normal_rat.max_mean,
        }
    elif cfg.latent_dist.type == "categorical":
        leaf_type = Categorical
        leaf_kwargs = {"num_bins": cfg.latent_dist.categorical.K}
    else:
        raise ValueError(f"Unknown distribution {cfg.latent_dist.type}.")
    return leaf_kwargs, leaf_type
