from typing import Dict

import gymnasium as gym
import ray.rllib.algorithms.dqn as dqn


class DQNTrainerGenerator:
    DEFAULT_CONFIG = dqn.DQNConfig().to_dict()
    DEFAULT_CONFIG["exploration_config"]['epsilon_timesteps'] = 10000
    DEFAULT_CONFIG["model"]['fcnet_hiddens'] = [512, 512, 256]
    DEFAULT_CONFIG["lr"] = 0.001
    DEFAULT_CONFIG["gamma"] = 0.99
    DEFAULT_CONFIG['train_batch_size'] = 256
    # number of samples to add to replay buffer
    DEFAULT_CONFIG['rollout_fragment_length'] = 1
    # control how often we sample (play in the environment) vs how often we update/train the worker
    # !!!WARNING does not work with version 1.13
    # TRAINING_FREQ = 4
    # DEFAULT_CONFIG['training_intensity'] = DEFAULT_CONFIG['train_batch_size'] // TRAINING_FREQ
    # control how often we update the target model
    DEFAULT_CONFIG["target_network_update_freq"] = 1000
    # !!!TODISCUSS distributed q-learning (approximates the distribution of the rewards instead)
    DEFAULT_CONFIG["num_atoms"] = 1
    DEFAULT_CONFIG["v_min"] = -1
    DEFAULT_CONFIG["v_max"] = 1
    # double DQN modification
    DEFAULT_CONFIG["double_q"] = True
    # !!!TODISCUSS prioritized replay modification (more recent observations in replay_buffer are chosen more often)
    # DEFAULT_CONFIG["prioritized_replay"] = True
    # DEFAULT_CONFIG["prioritized_replay_alpha"] = 0.6
    # DEFAULT_CONFIG["prioritized_replay_beta"] = 0.4
    # DEFAULT_CONFIG["beta_annealing_fraction"] = 0.2
    # DEFAULT_CONFIG["final_prioritized_replay_beta"] = 0.4
    # DEFAULT_CONFIG["prioritized_replay_eps"] = 1e-6
    # !!!TODISCUSS dueling learning
    # DEFAULT_CONFIG["dueling"] = True
    # multi-step learning (using n-steps in the future for loss function)
    DEFAULT_CONFIG["n_step"] = 1
    # !!!TODISCUSS use noisy parameters for exploration
    DEFAULT_CONFIG["noisy"] = False
    DEFAULT_CONFIG["sigma0"] = 0.5

    def get_default_config(self):
        return self.DEFAULT_CONFIG

    def generate_trainer(self, env: gym.Env, config: Dict):
        return dqn.DQN(env=env, config=dqn.DQNConfig.from_dict(config))
