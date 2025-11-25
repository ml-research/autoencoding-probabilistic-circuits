from apc.enums import ModelName
import sys
import logging
import os
import hydra
import lightning as L
from numpy import resize
import omegaconf
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import seed_everything
from torchinfo import summary
from time import time
from exp_utils import time_diff_to_str

from apc.train import train_autoencoder, train_iterative_imputer
from apc.utils import (
    preprocess_cfg,
    setup_rtpt,
    extract_classes_0_to_9,
    count_params,
    setup_model,
    determine_normalization,
    setup_logging,
)
from pickle import dump, load
from embeddings_2d import make_vis_embedding
from apc.exp_utils import catch_exception
from simple_einet.data import build_dataloader, is_1d_data, is_classification_data
from test import evaluate_model, test_reconstructions
from test_downstream_task import test_downstream_task
from torch.profiler import profile, record_function, ProfilerActivity

logger = logging.getLogger(__name__)

PRECISION = "bf16-mixed"


def main(cfg: DictConfig) -> None:
    """Main function."""

    #######################
    # Configuration Setup #
    #######################
    preprocess_cfg(cfg)
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir

    if cfg.debug:
        MAX_WORKERS = 0
    else:
        MAX_WORKERS = 8

    #################
    # Setup Logging #
    #################
    setup_logging(dir=run_dir)
    logger.info(f"Working directory: {os.getcwd()}")

    # Save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # Save run_dir in config
    with open_dict(cfg):
        cfg.run_dir = run_dir

    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Run dir: {run_dir}")

    ####################
    # Environment Setup #
    ####################

    seed_everything(cfg.seed, workers=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    devices = [0] if cfg.device == "cuda" else 1
    fabric = L.Fabric(accelerator=cfg.device, devices=devices, precision=cfg.precision)
    # fabric.launch()

    rtpt = setup_rtpt(cfg, iters=cfg.train.iters)
    rtpt.start()

    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    ###############
    # Setup Model #
    ###############
    lr_scheduler, warmup_scheduler, model, optimizer = setup_model(cfg)

    # Compile
    if cfg.compile:
        model = torch.compile(model)

    # Count parameters, adds to wandb and prints to log
    count_params(cfg_container, model)

    ##############
    # Setup Data #
    ##############

    hydra_cfg = HydraConfig.get()
    results_dir = hydra_cfg.runtime.output_dir
    os.makedirs(results_dir, exist_ok=True)

    normalize = determine_normalization(cfg)

    data_dir = cfg.data_dir
    train_loader, train_loader_no_loop, val_loader, test_loader = build_dataloader(
        dataset_name=cfg.dataset,
        batch_size=cfg.train.bs,
        data_dir=data_dir,
        num_workers=min(os.cpu_count(), MAX_WORKERS),
        normalize=normalize,
        loop=True,
        seed=cfg.seed,
    )

    train_loader, train_loader_no_loop, val_loader, test_loader = fabric.setup_dataloaders(
        train_loader, train_loader_no_loop, val_loader, test_loader
    )

    data_const = get_data_const(val_loader, cfg.dataset)

    # Print model
    logger.info("\n" + str(model))

    # Show summary
    if model.has_model_summary():
        model_summary = summary(
            model,
            input_data=data_const.to("cpu"),
            depth=3,
            col_names=["input_size", "output_size", "num_params", "params_percent"],
            verbose=0,
        )
        logger.info(model_summary)

    # To fabric
    model, optimizer = fabric.setup(model, optimizer)
    # Necessary for autocasting
    model.mark_forward_method("loss")
    model.mark_forward_method("encode")
    model.mark_forward_method("decode")
    model.mark_forward_method("sample_x")
    model.mark_forward_method("sample_z")
    model.mark_forward_method("reconstruct")

    ###############
    # Train Model #
    ###############

    model_ckpt_name = f"model.ckpt"
    model_ckpt_path = os.path.join(results_dir, model_ckpt_name)
    model_ckpt_state = {model_ckpt_name: model}

    # Initialize as late as possible, such that early errors will not create a new wandb run
    logger.info("Initializing WandB")
    run = wandb.init(
        project="apcs-2",
        config=cfg_container,
        name=cfg.tag,
        group=cfg.group,
        dir=run_dir,
        mode="online" if (not cfg.debug) or cfg.wandb else "disabled",
    )

    if cfg.load_and_eval:
        # Load model and just evaluate
        logger.info("Loading model")
        if cfg.model_name in [ModelName.MICE, ModelName.MISSFOREST]:
            with open(model_ckpt_path, "rb") as f:
                model = load(f)
        else:
            fabric.load(cfg.load_and_eval, model_ckpt_state)
    else:
        # Train model
        logger.info("Training model")
        t_start = time()
        if cfg.model_name in [ModelName.MICE, ModelName.MISSFOREST]:
            train_iterative_imputer(cfg, model, train_loader, val_loader, data_const, results_dir)
        else:
            train_autoencoder(
                cfg,
                fabric,
                rtpt,
                model,
                train_loader,
                val_loader,
                optimizer,
                lr_scheduler,
                warmup_scheduler,
                data_const,
                results_dir=results_dir,
            )
        logger.info(f"Training took {time_diff_to_str(t_start, time())}")

        # Save model
        logger.info("Saving model")
        if cfg.model_name in [ModelName.MICE, ModelName.MISSFOREST]:
            with open(model_ckpt_path, "wb") as f:
                dump(model, f, protocol=5)
        else:
            fabric.save(model_ckpt_path, model_ckpt_state)

    ##########################
    # Test on train/val/test #
    ##########################
    if cfg.test.rec:
        logger.info("Testing reconstructions")
        test_reconstructions(model, test_loader, tag="test", debug=cfg.debug, cfg=cfg, results_dir=results_dir)

    if cfg.test.tsne and is_classification_data(cfg.dataset) and model.has_embeddings():
        logger.info("Making t-SNE embeddings")
        make_vis_embedding(
            model, dataloader=val_loader, seed=cfg.seed, cfg=cfg, function="tsne", results_dir=results_dir
        )

    logger.info("Evaluating model")
    evaluate_model(model, test_loader, "test", cfg=cfg, results_dir=results_dir)

    ############################
    # Evaluate downstream task #
    ############################
    if cfg.test.downstream and is_classification_data(cfg.dataset) and model.has_embeddings():
        # Reinit rtpt
        rtpt = setup_rtpt(cfg, iters=cfg.downstream_task.train.iters, add_tag="_downstream")
        rtpt.start()

        logger.info("Evaluating downstream task")
        test_downstream_task(
            model_encoder=model,
            train_loader=train_loader_no_loop,
            val_loader=val_loader,
            test_loader=test_loader,
            cfg=cfg,
            rtpt=rtpt,
            fabric=fabric,
            results_dir=results_dir,
        )


def get_data_const(val_loader: torch.utils.data.DataLoader, dataset: str) -> torch.Tensor:
    """Get constant data for model summary and visualization."""
    if is_1d_data(dataset):
        return next(iter(val_loader))[0][:20]
    else:
        return extract_classes_0_to_9(val_loader, dataset=dataset, N=2)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    # Track exceptions
    exit_code = 0
    error = None

    try:
        # Run the main script
        main(cfg)
    except Exception as e:
        # Log exceptions
        logger.exception("An error occurred during execution")
        exit_code = 1
        error = e
    finally:
        logger.info("Finished. To rerun, use the following command:")
        logger.info(f"python {' '.join(sys.argv)}")
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)

    # If an error occurred, move the run directory to the error directory and raise the error
    if error is not None:
        catch_exception(output_directory=cfg.run_dir, e=error)
        raise error


if __name__ == "__main__":
    main_hydra()
