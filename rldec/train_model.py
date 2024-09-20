import json
import os
import datetime
import logging.config
from typing import Optional

from .callbacks import MonitoringCallback
from .agents.dqntrainer import DQNTrainerGenerator
from .loggers import TrainLogger
from .utils import combine_into
from .environments.sequential import SequentialEnv
from .environments.flattened import FlattenedEnv
from .data_handler import DataHandler
from .config import MODELS_PATH, DATA_PATH, SUPPORTED_FORMATS


ENVIRONMENT_NAME_MAP = {
    "combsequential": SequentialEnv,
    "sequential": SequentialEnv,
    "flattened": FlattenedEnv
}
DEFAULT_CONFIG_MAP = {
    "combsequential": "default_config_CombSequential.json",
    "sequential": "default_config_Sequential.json",
    "flattened": "default_config_Flattened.json"
}


def train_model(app_name: str, rldec_approach: str = "combsequential", features_name: str = "structural",
                n_iterations: int = 20, app_repo: Optional[str] = None, config_path: Optional[str] = None,
                data_path: str = DATA_PATH, output_path: str = MODELS_PATH, model_name: Optional[str] = None,
                eval_model: bool = False, granularity: str = "class", is_distributed: bool = False,
                n_episodes: Optional[int] = None, data_format: str = "parquet", *args, **kwargs):
    if not data_format in SUPPORTED_FORMATS:
        raise ValueError(f"Data format {data_format} is not supported")
    logger = logging.getLogger("Trainer")
    logger.info(f"training on {app_name} with {rldec_approach}")
    # get hyper_parameter grid
    trainer_created = False
    try:
        # setup environment hyper-parameters
        experiment_starting_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        experiment_name = model_name if model_name is not None else f"rldec_{rldec_approach}_{experiment_starting_time}"
        checkpoint_path = os.path.join(output_path, app_name, experiment_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        logging.config. fileConfig(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logging.conf'),
                                   disable_existing_loggers=False,
                                  defaults={
                                      "logfilename": os.path.join(checkpoint_path, "logs.log")
                                  })
        logger = logging.getLogger('Training')
        env = ENVIRONMENT_NAME_MAP[rldec_approach]
        # load default config
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "default_configs", DEFAULT_CONFIG_MAP[rldec_approach]), "r") as f:
            rldec_default_config = json.load(f)
        # load default ray config
        trainer_generator = DQNTrainerGenerator()
        trainer_config = trainer_generator.get_default_config()
        combine_into(rldec_default_config, trainer_config)
        # load user selected config
        if config_path is not None:
            with open(config_path, "r") as f:
                custom_config = json.load(f)
            combine_into(custom_config, trainer_config)
        # configure hyperparams
        # trainer_config["rldec_approach"] = rldec_approach
        # app_path = os.path.join(data_path, app_name)
        data_handler = DataHandler(app_name, data_path, app_repo, granularity=granularity, is_distributed=is_distributed,
                                   data_format=data_format, *args, **kwargs)
        app_path = data_handler.app_path
        data_handler.pull_all()
        names = data_handler.get_names()
        trainer_config["env_config"]["atoms"] = names
        if n_episodes is not None:
            if rldec_approach != "flattened":
                trainer_config["train_batch_size"] = 1
                trainer_config["min_sample_timesteps_per_iteration"] = len(names)
                trainer_config["batch_mode"] = "complete_episodes"
                # trainer_config["env_config"]["spec"] = {"max_episode_steps":len(names)}
                n_iterations = n_episodes
                trainer_config["timesteps_per_iteration"] = len(names)
                logger.debug("n_iterations set to {} and timesteps_per_iteration set to {}".format(
                    n_iterations, trainer_config["timesteps_per_iteration"]))
            else:
                logger.debug(f"Flattened approach does not support n_episodes parameter. Using {n_iterations} "
                             f"iterations instead with {trainer_config['timesteps_per_iteration']} steps")
        trainer_config["env_config"]["data_format"] = data_format
        if rldec_approach != "combsequential":
            trainer_config["env_config"]["features_name"] = features_name
        trainer_config["env_config"]["app_name"] = app_name
        trainer_config["env_config"]["data_path"] = app_path
        trainer_config["env_config"]["rldec_approach"] = rldec_approach
        trainer_config["env_config"]["n_iterations"] = n_iterations
        trainer_config["env_config"]["granularity"] = granularity
        trainer_config["env_config"]["data_format"] = data_format
        # setup logging config
        log_config = dict(
            type=lambda config, logdir, trial=None: TrainLogger(config, logdir, trial, trainer_config, dict()),
            logdir=os.path.join(output_path, app_name),
            experiment_name=experiment_name,
            experiment_file_id=experiment_starting_time,
            starting_time=experiment_starting_time,
        )
        trainer_config["logger_config"] = log_config
        # setup evaluation config
        if eval_model:
            eval_config = trainer_config["eval_config"]
            eval_config["interaction_data_path"] = data_handler.get_path(f"{granularity}_interactions.{data_format}")
            eval_config["semantic_data_path"] = data_handler.get_path(f"{granularity}_word_count.{data_format}")
        else:
            eval_config = dict(
                info_freq=-1,
                eval_freq=0,
                record_last=True
            )
        trainer_config.pop("eval_config")
        # add callbacks
        trainer_config["callbacks"] = lambda x=None: MonitoringCallback(x, eval_config, log_config)
        trainer_config["logger_config"] = log_config
        # initialize trainer
        # trainer_config["min_train_timesteps_per_iteration"] = trainer_config["timesteps_per_iteration"]
        # trainer_config["num_steps_sampled_before_learning_starts"] = 0
        trainer = trainer_generator.generate_trainer(env=env, config=trainer_config)
        # print(trainer.config.timesteps_per_iteration, trainer_config["timesteps_per_iteration"])
        # trainer.config.min_train_timesteps_per_iteration = trainer.config.timesteps_per_iteration
        trainer_created = True
        # start training
        logger.info("starting training")
        for i in range(n_iterations):
            logger.info("starting training iteration {}".format(i))
            result = trainer.train()
            logger.info("finished training iteration {}".format(i))
        logger.info("saving checkpoint")
        trainer.save(checkpoint_path)
        # logger.info("cleaning up trainer")
        # trainer.cleanup()
        logger.info("closing loggers")
        trainer._result_logger.close()
        logger.info("closing callbacks")
        trainer.workers.local_worker().callbacks.cleanup()
        # trainer.callbacks.cleanup()
    except Exception as e:
        if trainer_created:
            trainer._result_logger.close()
            trainer.workers.local_worker().callbacks.cleanup()
            trainer.callbacks.cleanup()
        raise e
    logger.info("finished")
