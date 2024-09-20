import json
import os
import logging.config
from typing import Optional, List, Dict

import numpy as np
from ray.rllib.policy.policy import Policy

from .environments.sequential import SequentialEnv
from .environments.flattened import FlattenedEnv
from .model_handler import ModelHandler


ENVIRONMENT_NAME_MAP = {
    "combsequential": SequentialEnv,
    "sequential": SequentialEnv,
    "flattened": FlattenedEnv
}


def vector_to_partitions(decomposition: np.ndarray, atoms: List[str]) -> Dict[str, List[str]]:
    partitions = dict()
    for partition_id in np.unique(decomposition):
        partitions[f"partition_{partition_id}"] = []
        for atom_id in np.where(decomposition == partition_id)[0]:
            partitions[f"partition_{partition_id}"].append(atoms[atom_id])
    return partitions


def generate_decomposition(app_name: str, models_path: str, model_name: Optional[str] = None, verbose: bool = True,
                           output_path: Optional[str] = None, num_episodes: int = 1, select_strategy: str = "last") -> (
        Dict)[str, List[str]]:
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        logging.config.fileConfig(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logging.conf'),
                                  disable_existing_loggers=False,
                                  defaults={
                                      "logfilename": os.path.join(output_path, "logs.log")
                                  })
    logger = logging.getLogger('Decomposer')
    model_handler = ModelHandler(models_path)
    # find model and most recent checkpoint
    model_path, model_name = model_handler.get_model(app_name, model_name)
    logger.info(f"decomposing {app_name} with {model_name}")
    # load environment
    env_config = model_handler.load_env_config(model_path)
    rldec_approach = env_config["rldec_approach"]
    env = ENVIRONMENT_NAME_MAP[rldec_approach]
    production = num_episodes == 1 or select_strategy == "last"
    env_instance = env(env_config, production=production)
    atoms = env_config["atoms"]
    # load model
    trainer = Policy.from_checkpoint(model_path)["default_policy"]
    # begin the decomposition
    logger.debug("beginning the decomposition")
    assert num_episodes > 0
    decompositions = []
    rewards = []
    for i in range(num_episodes):
        logger.debug(f"beginning episode {i}")
        episode_reward = 0
        done = False
        obs, _ = env_instance.reset()
        while not done:
            action, _, _ = trainer.compute_single_action(obs, explore=False)
            obs, reward, done, _, info = env_instance.step(action)
            episode_reward += reward
            if verbose:
                logger.debug(f"selected action {action} and received reward {reward}")
        logger.debug(f"finished episode {i} with cumulative reward {episode_reward}")
        decomposition = env_instance.process_state(obs)
        decompositions.append(decomposition)
        rewards.append(episode_reward)
    if select_strategy != "last" and not production:
        decompositions = [x for _, x in sorted(zip(rewards, decompositions), key=lambda x: x[0])]
    decomposition = decompositions[-1]
    if verbose:
        logger.info(decomposition)
    logger.info("finished the decomposition")
    # parse vector to partition map
    decomposition = vector_to_partitions(decomposition, atoms)
    if output_path is not None:
        with open(os.path.join(output_path, "decomposition.json"), "w") as f:
            json.dump(decomposition, f)
    return decomposition
