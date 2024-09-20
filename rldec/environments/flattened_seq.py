from typing import Dict

import gym
import numpy as np

from ..utils import extract_microservices
from . import DecompEnv


class FlattenedSeqEnv(DecompEnv):
    """
    Implementation of the Flattened approach in the rldec formulation.
    """
    def __init__(self, env_config: Dict):
        super().__init__(env_config)
        # load hyper-params
        self.max_unchanged_rewards = env_config["max_unchanged_rewards"]
        self.space_shape = (self.n_elements * (self.n_elements - 1)) // 2
        self.observation_space = gym.spaces.MultiBinary(self.space_shape)
        self.action_space = gym.spaces.Discrete(self.n_elements)
        # if "max_episode_steps" in env_config:
        #     self.max_episode_steps = env_config["max_episode_steps"]
        # else:
        #     self.max_episode_steps = self.space_shape // 2
        # set environment
        # self.reset(seed=, options=)

    def reset(self):
        self.current_step = 0
        self.prev_score = 0
        self.ep_reward = 0
        self.rewards = list()
        self.current_state_ = np.zeros(self.n_elements, dtype=np.int16)
        self.current_state_flattened = np.zeros(self.observation_space.n)
        return self.current_state

    def step(self, action):
        self.update(action)
        score = self.get_fitness(self.process_state(self.current_state))
        self.current_step += 1
        done = self.current_step >= self.max_episode_steps
        reward = score - self.prev_score
        self.rewards.append(reward)
        self.prev_score = score
        self.ep_reward += reward
        if len(self.rewards) > self.max_unchanged_rewards:
            done = np.equal(self.rewards[-self.max_unchanged_rewards:], self.rewards[-1]).all()
        info = dict(
            score=score
        )
        return self.current_state, reward, done, info

    def update(self, action):
        self.current_state[action] = 1 - self.current_state[action]

    def process_state(self, state):
        return extract_microservices(state, self.n_elements)

    def unflatten(self, flattened_obs):
        return flattened_obs
