# Autoencoding Probabilistic Circuits

<p align="center"><img src="res/figure-2.svg" width="90%"></p>

Code release for the paper **Tractable Representation Learning with Probabilistic Circuits**. The project implements autoencoding probabilistic circuits (APCs): a tractable PC encoder paired with a neural decoder, differentiable sampling (SIMPLE) for end-to-end training, and evaluation tooling for missing-data robustness, downstream tasks, and knowledge distillation from neural autoencoders.

The repository is organized around the `apc/` package and Hydra configs in `conf/`.

## Installation
- Python >= 3.10 and a recent PyTorch build (CUDA recommended for images).
- Clone the repo and install:
  ```bash
  pip install .
  ```
- Set data/result roots or rely on defaults (`~/data` and `~/results`):
  ```bash
  export DATA_DIR=/path/to/datasets
  export RESULTS_DIR=/path/to/outputs
  ```

## Quickstart
Train and evaluate an APC on MNIST-32 with the convolutional PC encoder and neural decoder:

```bash
python apc/main.py data=mnist-32 model=apc group=test
```

## Configuration Notes
- **Hydra configs:** default config is `conf/config.yaml`; datasets and model presets live under `conf/data/` and `conf/model/`.
- **Logging:** WandB is off by default (`wandb=false`); enable with your project name if desired. Logs and checkpoints are stored in `${RESULTS_DIR}/${dataset}/${group}/${subgroup}/${timestamp}_${tag}`.
- **Environment variables:** `DATA_DIR` and `RESULTS_DIR` override storage roots;

## Citation
If you use this code, please cite:

``` bibtex
TODO
```
