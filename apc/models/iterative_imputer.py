#!/usr/bin/env python3
import numpy as np
from simple_einet.data import is_1d_data
from sklearn.experimental import enable_iterative_imputer
from apc.enums import DecoderType
from apc.models.decoder.abstract_decoder import AbstractDecoder
import torch
from apc.models.encoder.abstract_encoder import AbstractEncoder
from apc.models.abstract_model import AbstractAutoencoder
from apc.models.encoder.nn_encoder import DummyEncoder
from apc.models.decoder.nn_decoder import DummyDecoder
from sklearn.impute import IterativeImputer as SklearnIterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class IterativeImputer(AbstractAutoencoder):
    """
    IterativeImputer is a class that implements an iterative imputer for missing data.
    It inherits from the AbstractAutoencoder class and uses its methods and properties.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Construct base estimator
        match cfg.iterative_imputer.estimator:
            case "bayesianridge":
                estimator = BayesianRidge(max_iter=cfg.iterative_imputer.br.max_iter)
            case "randomforest":
                estimator = RandomForestRegressor(
                    n_estimators=cfg.iterative_imputer.rf.n_estimators, n_jobs=cfg.iterative_imputer.rf.n_jobs, max_depth=cfg.iterative_imputer.rf.depth
                )

            case _:
                raise ValueError(
                    f"Unknown estimator: {cfg.iterative_imputer.estimator}, must be one of: bayesridge, randomforest"
                )

        # Determine the imputer parameters
        is_image_data = not is_1d_data(cfg.dataset)
        if is_image_data:
            min_value = -1.0
            max_value = 1.0
            n_nearest_features = np.round(np.sqrt(self.data_shape.num_pixels * self.data_shape.channels)).astype(int)
        else:
            min_value = -np.inf
            max_value = np.inf
            n_nearest_features = None

        # Construct sklearn iterative imputer class
        self.imputer = SklearnIterativeImputer(
            estimator=estimator,
            skip_complete=self.cfg.iterative_imputer.skip_complete,
            verbose=0,
            min_value=min_value,
            max_value=max_value,
            n_nearest_features=n_nearest_features,
        )

        # Add dummy paramter to infer device (no grad)
        self.__dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def has_model_summary(self):
        return False

    def has_embeddings(self):
        return False

    def fit(self, x: torch.Tensor) -> None:
        """
        Fit the model to the input data.

        Args:
            x (torch.Tensor): Input tensor.
        """
        x = x.view(x.shape[0], -1).cpu().numpy()
        self.imputer.fit(x)

    def make_encoder(self) -> AbstractEncoder:
        return DummyEncoder(self.cfg)

    def make_decoder(self) -> AbstractDecoder:
        return DummyDecoder(self.cfg)

    def sample_z(
        self,
        num_samples: int,
        seed: int,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample from the latent space.

        Args:
            num_samples (int): Number of samples to generate.
            seed (int): Seed for random number generation.
            tau (float, optional): Temperature parameter for sampling. Defaults to 1.0.

        Returns:
            torch.Tensor: Sampled latent variables.
        """
        return torch.randn(num_samples, self.latent_dim, device=self.mydevice)

    def reconstruct(self, x: torch.Tensor, decoder_type: DecoderType = None, fill_evidence=False) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        mask = x.isnan()
        y = self.imputer.transform(x.cpu().numpy())
        y = torch.from_numpy(y).to(x.device)
        if mask.any() and fill_evidence:
            y[~mask] = x[~mask]
        return y.view(x.shape[0], *self.data_shape)

    def loss(self, x) -> dict[str, torch.Tensor]:
        return {"rec": torch.zeros(1, device=self.mydevice)}
