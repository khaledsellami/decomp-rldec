from typing import Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from ..utils import extract_microservices
from . import DecompEnv


class FlattenedEnv(DecompEnv):
    """
    Implementation of the Flattened approach in the rldec formulation.
    """
    def __init__(self, env_config: Dict, production: bool = False):
        super().__init__(env_config, production)
        # load hyper-params
        self.max_unchanged_rewards = env_config["max_unchanged_rewards"]
        self.space_shape = (self.n_elements * (self.n_elements - 1)) // 2
        self.observation_space = gym.spaces.MultiBinary(self.space_shape)
        self.action_space = gym.spaces.Discrete(self.space_shape)
        if "max_episode_steps" in env_config:
            self.max_episode_steps = env_config["max_episode_steps"]
        else:
            self.max_episode_steps = self.space_shape // 2
        # set environment
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.current_step = 0
        self.prev_score = 0
        self.ep_reward = 0
        self.rewards = list()
        self.current_state = np.zeros(self.observation_space.n, dtype=bool)
        return self.current_state, dict()

    def step(self, action):
        self.update(action)
        self.current_step += 1
        done = self.current_step >= self.max_episode_steps
        if self.production:
            info = dict()
            return self.current_state, np.nan, done, False, info
        score = self.get_fitness(self.process_state(self.current_state))
        reward = score - self.prev_score
        self.rewards.append(reward)
        self.prev_score = score
        self.ep_reward += reward
        if len(self.rewards) > self.max_unchanged_rewards:
            done = np.equal(self.rewards[-self.max_unchanged_rewards:], self.rewards[-1]).all()
        info = dict(
            score=score
        )
        return self.current_state, reward, done, False, info

    def update(self, action):
        self.current_state[action] = not self.current_state[action]

    def process_state(self, state):
        return extract_microservices(state, self.n_elements)

    def unflatten(self, flattened_obs):
        return flattened_obs
