import datetime
import json
import logging
import os
from typing import Optional, Dict

from ray.tune.logger import JsonLogger
from ray.air.constants import EXPR_RESULT_FILE
from ray.tune.utils.util import SafeFallbackEncoder

from .utils import generate_id


TRAIN_CONFIG_NAME = "train_config"
ENV_CONFIG_NAME = "env_config"
METRICS_NAME = "episode_results"
DEFAULT_NAME = "DefaultName"


class TrainLogger(JsonLogger):

    def __init__(self, config: Dict, logdir: str, trial: Optional["Trial"] = None, train_config: Dict = {},
                 to_exclude_train_config: Optional[Dict] = None):
        self.train_config = train_config
        if to_exclude_train_config is None:
            self.to_exclude_train_config = {"env_config": ["features", "data_path"]}
        else:
            self.to_exclude_train_config = to_exclude_train_config
        super().__init__(config, logdir, trial)

    def _init(self):
        self.logger = logging.getLogger("Callback")
        if "experiment_file_id" not in self.config:
            self.experiment_file_id = generate_id(4)
        else:
            self.experiment_file_id = self.config["experiment_file_id"]
        if "starting_time" not in self.config:
            self.starting_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        else:
            self.starting_time = self.config["starting_time"]
        if "experiment_name" not in self.config:
            self.experiment_name = self.starting_time + DEFAULT_NAME + self.experiment_file_id
        else:
            self.experiment_name = self.config["experiment_name"]
        if not os.path.exists(os.path.join(self.logdir, self.experiment_name)):
            os.mkdir(os.path.join(self.logdir, self.experiment_name))
        self.logdir = os.path.join(self.logdir, self.experiment_name)
        self.update_train_config(self.train_config)
        local_file = os.path.join(self.logdir, EXPR_RESULT_FILE)
        self.initial_result = True
        self.local_out = open(local_file, "a")
        self.write("[")
        self.closed = False
        self.logger.debug("recording experiment information in {}".format(self.logdir))

    def update_train_config(self, train_config: Dict):
        # to_exclude = {"env_config": ["features", "data_path"]}
        self.train_config = train_config
        config_to_save = self.train_config.copy()
        for param in self.to_exclude_train_config:
            if param in config_to_save:
                if isinstance(self.to_exclude_train_config[param], list):
                    config_to_save[param] = {
                        k: v for k, v in config_to_save[param].items() if k not in self.to_exclude_train_config[param]
                    }
                else:
                    config_to_save.pop(param)
        config_to_save["logger_config"]["experiment_id"] = self.experiment_file_id
        config_to_save["logger_config"]["starting_time"] = self.starting_time
        config_out = os.path.join(self.logdir, "{}.json".format(TRAIN_CONFIG_NAME))
        with open(config_out, "w") as f:
            json.dump(config_to_save, f, indent=2, sort_keys=True, cls=SafeFallbackEncoder)
        env_config_out = os.path.join(self.logdir, "{}.json".format(ENV_CONFIG_NAME))
        with open(env_config_out, "w") as f:
            json.dump(config_to_save["env_config"], f, indent=2,
                      sort_keys=True, cls=SafeFallbackEncoder)


    def on_result(self, result: Dict):
        to_exclude = {
            "sampler_results": "sampler_results",
            "env_config": "env_config",
            "config": "config",
        }
        result = result.copy()
        for param in to_exclude:
            if param in result:
                if isinstance(to_exclude[param], list):
                    result[param] = {
                        k: v for k, v in result[param].items() if k not in to_exclude[param]
                    }
                else:
                    result.pop(param)
        for k in ["custom_metrics", "hist_stats"]:
            if k in result:
                for m in result[k]:
                    if m in result[k]:
                        result[k][m] = result[k][m][-result["episodes_this_iter"]:]
        if self.initial_result:
            self.initial_result = False
        else:
            self.write(",\n")
        json.dump(result, self, cls=SafeFallbackEncoder)
        self.local_out.flush()

    def close(self):
        assert not self.closed
        self.logger.debug("closing logger and all files")
        self.write("]")
        self.local_out.flush()
        self.local_out.close()


class ResultLogger:
    def __init__(self, config, logdir, filename=METRICS_NAME):
        self.config = config
        self.logdir = logdir
        self.filename = filename
        self.logger = logging.getLogger("Callback")
        if "experiment_file_id" not in self.config:
            self.experiment_file_id = generate_id(4)
        else:
            self.experiment_file_id = self.config["experiment_file_id"]
        if "starting_time" not in self.config:
            self.starting_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        else:
            self.starting_time = self.config["starting_time"]
        if "experiment_name" not in self.config:
            self.experiment_name = self.starting_time + DEFAULT_NAME + self.experiment_file_id
        else:
            self.experiment_name = self.config["experiment_name"]
        if not os.path.exists(os.path.join(self.logdir, self.experiment_name)):
            os.mkdir(os.path.join(self.logdir, self.experiment_name))
        self.logdir = os.path.join(self.logdir, self.experiment_name)
        local_file = os.path.join(self.logdir, self.filename + ".json")
        self.initial_result = True
        self.local_out = open(local_file, "a")
        # self.local_out.write("[")
        self.closed = False
        self.additional_files = dict()
        self.additional_files_init = dict()
        self.logger.debug("recording result information in {}".format(self.logdir))

    def update_on(self, result):
        if self.initial_result:
            self.initial_result = False
            self.local_out.write("[")
        else:
            self.local_out.write(",\n")
        json.dump(result, self.local_out, cls=SafeFallbackEncoder)
        self.local_out.flush()

    def close(self):
        assert not self.closed
        self.logger.debug("closing result logger")
        if not self.initial_result:
            self.local_out.write("]")
            self.local_out.flush()
        self.local_out.close()
        self.closed = True

