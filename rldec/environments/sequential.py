from typing import Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from . import DecompEnv


class SequentialEnv(DecompEnv):
    """
    Implementation of the Sequential approach in the rldec formulation.
    """
    def __init__(self, env_config: Dict, production: bool = False):
        super().__init__(env_config, production)
        # load hyper-params
        self.max_microservices = env_config["max_microservices"]
        self.consider_outliers = env_config["consider_outliers"]
        assert isinstance(self.consider_outliers, bool)
        # self.space_shape = [
        #     gym.spaces.Discrete(self.max_microservices + 1 + self.consider_outliers) for i in range(self.n_elements)
        # ]
        self.space_shape = np.full(self.n_elements, self.max_microservices + 1 + self.consider_outliers)
        # self.observation_space = gym.spaces.Tuple(self.space_shape)
        self.observation_space = gym.spaces.MultiDiscrete(self.space_shape)
        self.action_space = gym.spaces.Discrete(self.max_microservices + self.consider_outliers)
        # set environment
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.current_step = 0
        self.current_state = np.zeros(self.n_elements, dtype=np.int16)
        # return self.current_state
        return self.current_state, dict()

    def step(self, action):
        action += 1
        self.update(action)
        self.current_step += 1
        done = self.current_step >= self.n_elements
        if self.production:
            info = dict()
            return self.current_state, np.nan, done, False, info
        reward = self.get_fitness(self.process_state(self.current_state)) if done else 0
        score = reward
        info = dict(
            score=score
        )
        return self.current_state, reward, done, False, info

    def update(self, action):
        self.current_state[self.current_step] = action

    def process_state(self, state):
        return state - 1 - self.consider_outliers

    def unflatten(self, flattened_obs):
        if flattened_obs.shape == self.current_state.shape:
            return flattened_obs
        num_classes = self.max_microservices + 1 + self.consider_outliers
        return tuple(
            [np.argmax(
                flattened_obs[i * num_classes:i * num_classes + num_classes]
            ) for i in range(self.n_elements)]
        )
