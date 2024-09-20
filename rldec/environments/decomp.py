from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from ..metrics import coverage, non_extreme_distribution
from ..features import load_features


class DecompEnv(gym.Env):
    """
    Abstract class for implementing the rldec Env. Contains the shared properties of the different potential
    formulations. Implements a OpenAI Gym Environment.
    """
    def __init__(self, env_config: Dict, production: bool = False):
        # get hyper-parameters
        self.production = production
        self.app_name = env_config["app_name"]
        self.features_name = env_config["features_name"]
        self.exclude_outliers = env_config["exclude_outliers"]
        self.use_coverage = env_config["use_coverage"]
        self.use_ned = env_config["use_ned"]
        self.atoms = env_config["atoms"]
        self.data_format = env_config["data_format"] if "data_format" in env_config else "npy"
        if "force_features" in env_config:
            self.force_features = env_config["force_features"]
        else:
            self.force_features = None
        self.granularity = env_config["granularity"] if "granularity" in env_config else "class"
        # load features
        self.n_elements = len(self.atoms)
        if self.production:
            self.features = None
            self.multi_feature = False
        else:
            self.data_path = env_config["data_path"]
            if isinstance(self.features_name, str):
                self.features = load_features(self.features_name, self.data_path, self.atoms, self.force_features,
                                              granularity=self.granularity, data_format=self.data_format)
                self.multi_feature = False
            else:
                assert isinstance(self.features_name, list)
                self.features = [
                    load_features(features_name, self.data_path, self.atoms, granularity=self.granularity,
                                  data_format=self.data_format)
                    for features_name in self.features_name
                ]
                self.multi_feature = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def update(self, action):
        raise NotImplementedError

    def get_fitness(self, microservices: np.ndarray) -> float:
        """
        calculates the fitness value of the current state.
        :param microservices: the array representation of the microservices decomposition
        :return: fitness values
        """
        if self.production:
            return np.nan
        if self.multi_feature:
            fitness = np.sum(
                [features.vectorized_modularity(microservices, self.exclude_outliers) for features in self.features]
            )
        else:
            fitness = self.features.vectorized_modularity(microservices, self.exclude_outliers)
        if self.use_coverage:
            fitness *= coverage(microservices)
        if self.use_ned:
            fitness *= 1 - non_extreme_distribution(microservices, exclude_outliers=self.exclude_outliers)
        return fitness

    def process_state(self, state):
        raise NotImplementedError

    def unflatten(self, flattened_obs):
        raise NotImplementedError
