import json
import torch
import tqdm
import wandb
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

from apc.enums import ModelName
from apc.utils import auc

from apc.models.abstract_model import AbstractAutoencoder, AbstractPcAutoencoder
from simple_einet.data import is_1d_data

from apc.losses import ssim, ms_ssim_custom
from pytorch_msssim import ssim, ms_ssim
import logging

logger = logging.getLogger(__name__)


def _mse(rec: torch.Tensor, data: torch.Tensor, cfg: DictConfig, is_image_data) -> float:
    """
    Compute the Mean Squared Error (MSE).

    Parameters:
        rec: torch.Tensor - Reconstructed data
        data: torch.Tensor - Original input data

    Returns:
        float - Mean Squared Error
    """
    data, rec = _scale(data, rec, is_image_data, cfg)
    error = F.mse_loss(rec, data, reduction="sum") / data.shape[0]
    return error.item()


def _bce(rec: torch.Tensor, data: torch.Tensor, cfg: DictConfig, is_image_data) -> float:
    """
    Compute the Binary Cross Entropy (BCE).

    Parameters:
        rec: torch.Tensor - Reconstructed data
        data: torch.Tensor - Original input data

    Returns:
        float - Binary Cross Entropy
    """
    data, rec = _scale(data, rec, is_image_data, cfg)
    error = F.binary_cross_entropy(rec, data, reduction="sum") / data.shape[0]
    return error.item()


def _scale(data, rec, is_image_data, cfg):
    if cfg.model_name == ModelName.APC:
        # if this is image data, normalize the data to [0, 1]
        if is_image_data:
            normalizer = 2**cfg.n_bits - 1
            rec = rec / normalizer
            data = data / normalizer
        else:
            # Noop
            pass

    elif cfg.model_name in [ModelName.VAE, ModelName.AE, ModelName.VAEM, ModelName.MIWAE, ModelName.HIVAE]:
        # NNs have (-1, 1) data
        if is_image_data:
            rec = (rec + 1) / 2
            data = (data + 1) / 2
        else:
            # Noop
            pass
    elif cfg.model_name in [ModelName.MICE, ModelName.MISSFOREST]:
        # MICE and MissForest have [0, 1] data
        if is_image_data:
            data = (data + 1) / 2

            # Scale rec to [0, 1]
            rec = (rec - rec.min()) / (rec.max() - rec.min())
        else:
            # Noop
            pass

    assert rec.shape == data.shape, f"Shapes don't match: {rec.shape} vs {data.shape}"
    if is_image_data:
        assert rec.min() >= 0 and rec.max() <= 1, f"Reconstructed data not in [0, 1]: {rec.min()} {rec.max()}"
        assert data.min() >= 0 and data.max() <= 1, f"Original data not in [0, 1]: {data.min()} {data.max()}"
    return data, rec


def _ssim(rec: torch.Tensor, data: torch.Tensor, cfg: DictConfig, is_image_data) -> float:
    data, rec = _scale(data, rec, is_image_data, cfg)
    return ssim(rec, data, data_range=1.0, size_average=True).item()


@torch.inference_mode()
def test_reconstructions(
    model: AbstractAutoencoder,
    loader: DataLoader,
    tag: str,
    iteration: int | None = None,
    debug=False,
    cfg: DictConfig = None,
    results_dir: str = None,
    prefix: str = "",
):
    """Test model on reconstruction error."""

    # Init dict to store test results
    results = {
        "mse": {dt: defaultdict(float) for dt in model.decoder_types()},
        "bce": {dt: defaultdict(float) for dt in model.decoder_types()},
        "ssim": {dt: defaultdict(float) for dt in model.decoder_types()},
    }

    ps = list(np.arange(0, 100, 5))

    is_image_data = not is_1d_data(cfg.dataset)

    count = 0

    # Count number of batches in loader
    n_batches = len(loader)

    for i, (data, _) in enumerate(tqdm.tqdm(loader, leave=False, desc="Testing")):
        count += 1
        if count > (n_batches * cfg.test.p_subset / 100):
            logger.info(f"Stopping test early at {cfg.test.p_subset}% of the dataset")
            break
        for dt in model.decoder_types():
            # Clone data to avoid inplace operations
            data = data.float()  # Needed since we put nan there later

            #######################
            # Full reconstruction #
            #######################
            x_rec = model.reconstruct(data, decoder_type=dt)

            results["mse"][dt]["full"] += _mse(x_rec, data, cfg, is_image_data)
            if is_image_data:
                results["bce"][dt]["full"] += _bce(x_rec, data, cfg, is_image_data)
                results["ssim"][dt]["full"] += _ssim(x_rec, data, cfg, is_image_data)

            if not is_1d_data(cfg.dataset):  # Don't do this for 1D data
                #############################
                # Right half reconstruction #
                #############################
                # Hold out right half of the images
                mask = torch.zeros_like(data).bool()
                mask[..., data.shape[3] // 2 :] = True
                data_cloned = data.clone()
                data_cloned[mask] = torch.tensor(float("nan"))
                rec = model.reconstruct(data_cloned, decoder_type=dt)
                rec[~mask] = data[~mask]  # fill in evidence
                results["mse"][dt]["right"] += _mse(rec, data, cfg, is_image_data)
                if is_image_data:
                    results["bce"][dt]["right"] += _bce(rec, data, cfg, is_image_data)
                    results["ssim"][dt]["right"] += _ssim(rec, data, cfg, is_image_data)

                #############################
                # Lower half reconstruction #
                #############################
                mask = torch.zeros_like(data).bool()
                mask[..., data.shape[2] // 2 :, :] = True
                data_cloned = data.clone()
                data_cloned[mask] = torch.tensor(float("nan"))
                rec = model.reconstruct(data_cloned, decoder_type=dt)
                rec[~mask] = data[~mask]  # fill in evidence
                results["mse"][dt]["lower"] += _mse(rec, data, cfg, is_image_data)
                if is_image_data:
                    results["bce"][dt]["lower"] += _bce(rec, data, cfg, is_image_data)
                    results["ssim"][dt]["lower"] += _ssim(rec, data, cfg, is_image_data)

            ######################
            # p % Reconstruction #
            ######################
            # Use the same random tensor as base for the mask such that p1 < p2 means mask1 is a subset of mask2
            gen = torch.Generator(device=data.device)
            gen.manual_seed(cfg.seed + i)

            # rand = torch.rand(data.shape[0], 1, data.shape[2], data.shape[3], generator=gen, device=data.device)
            # *Alternative*: make linspace from 0 to 1 by number of pixels and then shuffle
            # This ensures, that with every step in ps, we get an equal increase in the number of pixels missing
            linspace = torch.linspace(0, 1, data.shape[2] * data.shape[3], device=data.device)
            rand = torch.rand(data.shape[0], linspace.shape[0], device=data.device, generator=gen)
            perm_indices = torch.argsort(rand, dim=1)
            rand = linspace[perm_indices].view(data.shape[0], 1, data.shape[2], data.shape[3])

            for p in ps:
                mask = rand < (p / 100)

                # Repeat mask at dim=1 exactly data.shape[1] times to ensure that the full pixel is selected
                mask = mask.repeat(1, data.shape[1], 1, 1)
                data_cloned = data.clone()
                data_cloned[mask] = torch.tensor(float("nan"))
                rec = model.reconstruct(data_cloned, decoder_type=dt)
                rec[~mask] = data[~mask]
                results["mse"][dt][f"{p}"] += _mse(rec, data, cfg, is_image_data)
                if is_image_data:
                    results["bce"][dt][f"{p}"] += _bce(rec, data, cfg, is_image_data)
                    results["ssim"][dt][f"{p}"] += _ssim(rec, data, cfg, is_image_data)

        if debug:
            break

    # Normalize by count
    for metric, test_dict in results.items():
        for dt, metric_dict in test_dict.items():
            for key, value in metric_dict.items():
                results[metric][dt][key] = value / count

    # Collect everything we want to save
    save_dict = {}
    save_dict.update(results)

    for metric, test_dict in results.items():
        # Skip ssim for non-image data
        if (not is_image_data) and (metric == "ssim"):
            continue

        # Save full reconstruction error
        for dt, metric_dict in test_dict.items():
            wandb.log({f"{prefix}{tag}/{metric}/rec-{dt}": metric_dict["full"]}, commit=False)
            save_dict.update({f"{prefix}{tag}/{metric}/rec-{dt}": metric_dict["full"]})

        wandb_dict = {}

        # Reconstruction error percentages plot
        keys = [str(dt) for dt in model.decoder_types()]
        ys = []
        for dt in model.decoder_types():
            y = [test_dict[dt][f"{p}"] for p in ps]
            ys.append(y)

            # Add AUC
            auc_rec = auc(ps, y)
            wandb_dict.update({f"{prefix}{tag}/{metric}/auc-rec-{dt}": auc_rec})
            save_dict.update({f"{prefix}{tag}/{metric}/auc-rec-{dt}": auc_rec})

        # Add plot
        wandb_dict.update(
            {
                f"{prefix}{tag}/{metric}/reconstruction-plot": wandb.plot.line_series(
                    xs=[p for p in ps],
                    ys=ys,
                    keys=keys,
                    title=f"Rec. {metric.upper()} with missing Data",
                    xname="percentage",
                )
            }
        )

        if not is_1d_data(cfg.dataset):
            # Partial image holdout bar plot
            keys = []
            values = []
            for part in ["lower", "right", "full"]:
                for dt in model.decoder_types():
                    keys.append(f"{part}-{dt}")

            data = [[name, value] for (name, value) in zip(keys, values)]
            table = wandb.Table(data=data, columns=["part", metric])
            wandb_dict.update(
                {
                    f"{prefix}{tag}/{metric}/holdout-plot": wandb.plot.bar(
                        table, "part", metric, title=f"{metric.upper()} with Partial Image"
                    )
                }
            )

        wandb.log(wandb_dict, commit=True)

    # Save as json
    with open(f"{results_dir}/{prefix.replace('/', '-')}test-rec-images-{tag}.json", "w") as f:
        json.dump(save_dict, f)


@torch.inference_mode()
def evaluate_model(
    model: AbstractAutoencoder,
    loader: DataLoader,
    tag: str,
    iteration: int | None = None,
    cfg: DictConfig = None,
    results_dir: str = None,
    prefix: str = "",
):
    count = 0

    metrics = defaultdict(float)
    is_image_data = not is_1d_data(cfg.dataset)

    for data, _ in tqdm.tqdm(loader, leave=False):
        count += 1

        if hasattr(model, "log_likelihood"):
            # Log-likelihood
            lls = model.log_likelihood(x=data)
            metrics[f"{prefix}eval/{tag}/nll"] += -1 * lls.mean().item()
            metrics[f"{prefix}eval/{tag}/bpd"] += calculate_bits_per_dimension(
                lls, reduction="mean", data_shape=data.shape
            ).item()

        # We already do this in test_rec_images
        # Reconstruction
        for dt in model.decoder_types():
            x_rec = model.reconstruct(data, decoder_type=dt)
            metrics[f"{prefix}eval/{tag}/mse/rec-{dt}"] += _mse(x_rec, data, cfg, is_image_data)
            if is_image_data:
                metrics[f"{prefix}eval/{tag}/bce/rec-{dt}"] += _bce(x_rec, data, cfg, is_image_data)
                metrics[f"{prefix}eval/{tag}/ssim/rec-{dt}"] += _ssim(x_rec, data, cfg, is_image_data)

        if cfg.debug:
            break

    # Normalize
    metrics = {key: value / count for key, value in metrics.items()}
    metrics["iteration"] = iteration

    # Save as json
    with open(f"{results_dir}/{prefix.replace('/', '-')}eval-{tag}.json", "w") as f:
        json.dump(metrics, f)

    wandb.log(metrics, commit=True)


@torch.inference_mode()
def validation_loss(
    model: AbstractAutoencoder, loader: DataLoader, tag: str, iteration: int | None = None, cfg: DictConfig = None
):
    count = 0

    metrics = defaultdict(float)

    for data, _ in tqdm.tqdm(loader, leave=False):
        count += 1
        losses = model.loss(data)
        for key, value in losses.items():
            metrics[f"{tag}/{key}"] += value.item()

        if cfg.debug:
            break

    # Normalize
    metrics = {key: value / count for key, value in metrics.items()}
    metrics["iteration"] = iteration

    wandb.log(metrics, commit=True)
    return metrics


def calculate_bits_per_dimension(lls, data_shape, reduction="mean") -> torch.Tensor:
    # Change of base (natural log to base 2)
    log2_likelihood = lls / torch.log(torch.tensor(2.0))

    # Calculate bits per dimension (sum over all dimensions except batch)
    num_dimensions = np.prod(data_shape[1:])
    bits_per_dim = -(log2_likelihood.squeeze(-1) / num_dimensions)

    if reduction == "mean":
        return bits_per_dim.mean()
    elif reduction == "sum":
        return bits_per_dim.sum()
    else:
        raise ValueError(f"Reduction {reduction} not supported")
