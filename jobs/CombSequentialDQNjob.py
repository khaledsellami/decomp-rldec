import json
import os
import sys
import datetime
import logging.config

from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform

from rldec.callbacks import MonitoringCallback, ALLOWED_METRICS
from rldec.environments.sequential import SequentialEnv
from rldec.agents.dqntrainer import DQNTrainerGenerator
from rldec.loggers import TrainLogger
from rldec.data_handler import DataHandler
from user_config import DATA_PATH

if __name__ == "__main__":
    # Parsing input
    if len(sys.argv) != 4:
        sys.exit("Incorrect arguments received. Aborting!")
    else:
        app_name = sys.argv[1]
        job_num = int(sys.argv[2])
        job_id = sys.argv[3]
    # get hyper_parameter grid
    random_search_seed = 42
    n_RS_iterations = 15
    param_grid = dict(
        lr=uniform(loc=0.0001, scale=0.0009)
    )
    param_list = list(ParameterSampler(param_grid, n_iter=n_RS_iterations, random_state=random_search_seed))
    job_params = param_list[job_num]
    trainer_created = False
    granularity = "class"
    data_format = "parquet"
    try:
        # setup environment hyper-parameters
        APP_DATA_PATH = DATA_PATH # os.path.join(os.curdir, "data")
        APP = app_name
        features_name = ["structural", "semantic"]
        APP_PATH = os.path.join(APP_DATA_PATH, APP.lower())
        OUTPUT_PATH = os.path.join(os.path.curdir, "logs", APP.lower())
        EXPERIMENT_NAME = "CombSequentialDQN" + str(job_id)
        experiment_file_id = str(job_num)
        experiment_starting_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        experiment_name = "_".join([EXPERIMENT_NAME, experiment_file_id])
        checkpoint_path = os.path.join(OUTPUT_PATH, experiment_name)
        data_handler = DataHandler(app_name, DATA_PATH, None)
        app_path = data_handler.app_path
        data_handler.pull_all()
        class_names = data_handler.get_names()
        os.makedirs(checkpoint_path, exist_ok=True)
        logging.config. fileConfig(os.path.join(os.curdir, 'rldec', 'logging.conf'), disable_existing_loggers=False,
                                  defaults={
                                      "logfilename": os.path.join(checkpoint_path, "logs.log")
                                  })
        logger = logging.getLogger('Training')
        env = SequentialEnv
        env_config = dict(
            features_name=features_name,
            app_name=APP,
            data_path=APP_PATH,
            max_microservices=20,
            consider_outliers=False,
            exclude_outliers=False,
            use_coverage=False,
            use_ned=False,
            atoms=class_names,
            granularity=granularity,
            data_format=data_format,
        )
        # setup agent hyper-parameters
        trainer_generator = DQNTrainerGenerator()
        trainer_config = trainer_generator.get_default_config()
        trainer_config["model"]['fcnet_hiddens'] = [256, 512, 1024, 1024, 512, 256]
        trainer_config["lr"] = job_params["lr"]
        trainer_config['train_batch_size'] = 512
        trainer_config["target_network_update_freq"] = 500
        trainer_config["double_q"] = True
        trainer_config["n_step"] = 1
        trainer_config["dueling"] = True
        # setup logging config
        log_config = dict(
            type=lambda config, logdir, trial=None: TrainLogger(config, logdir, trial, trainer_config),
            logdir=OUTPUT_PATH,
            experiment_name=experiment_name,
            experiment_file_id=experiment_file_id,
            starting_time=experiment_starting_time,
        )
        # setup evaluation config
        eval_config = dict(
            info_freq=-1,
            eval_freq=-1,
            record_last=True,
            interaction_data_path=data_handler.get_path(f"{granularity}_interactions.{data_format}"),
            semantic_data_path=data_handler.get_path(f"{granularity}_word_count.{data_format}"),
            # metrics=ALLOWED_METRICS,
        )
        # setup global training hyper-parameters
        N_ITERATIONS = 100
        trainer_config["log_level"] = "INFO"
        trainer_config["framework"] = "torch"
        trainer_config["num_workers"] = 0  # 1 will force rllib to use trainer as the worker as well
        trainer_config["preprocessor_pref"] = None
        trainer_config["num_cpus_per_worker"] = 1
        trainer_config["num_cpus_for_driver"] = 1
        trainer_config["num_gpus"] = 1
        trainer_config["timesteps_per_iteration"] = 5000
        # trainer_config["keep_per_episode_custom_metrics"] = True
        # trainer_config["batch_mode"] = "truncate_episodes"
        trainer_config["env_config"] = env_config
        trainer_config["callbacks"] = lambda x=None: MonitoringCallback(x, eval_config, log_config)
        trainer_config["logger_config"] = log_config
        # initialize trainer
        trainer = trainer_generator.generate_trainer(env=env, config=trainer_config)
        trainer_created = True
        # start training
        logger.info("starting training")
        for i in range(N_ITERATIONS):
            logger.info("starting training iteration {}".format(i))
            result = trainer.train()
            logger.info("finished training iteration {}".format(i))
            if i % 20 == 0:
                logger.info("saving checkpoint")
                trainer.save(checkpoint_path)
        logger.info("saving checkpoint")
        trainer.save(checkpoint_path)
        # logger.info("cleaning up trainer")
        # trainer.cleanup()
        logger.info("closing loggers")
        trainer._result_logger.close()
        logger.info("closing callbacks")
        trainer.workers.local_worker().callbacks.cleanup()
        trainer.callbacks.cleanup()
    except Exception as e:
        if trainer_created:
            trainer._result_logger.close()
            trainer.workers.local_worker().callbacks.cleanup()
            trainer.callbacks.cleanup()
        raise e
    logger.info("finished")