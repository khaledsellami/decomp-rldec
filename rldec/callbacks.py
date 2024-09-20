from typing import Optional
import logging

import numpy as np
import pandas as pd
from ray.rllib import BaseEnv, RolloutWorker, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID, Dict
from sklearn.preprocessing import OneHotEncoder

from .loggers import ResultLogger
from .metrics import process_outliers, preprocessed_modularity, preprocessed_inter_call_percentage, \
    preprocessed_interface_number, preprocessed_non_extreme_distribution, coverage
from .utils import load_data


ALLOWED_METRICS = ["smq", "cmq", "icp", "ifn", "ned", "cov", "msn", "n_links"]
DEFAULT_METRICS = ["smq", "cmq", "icp", "ifn", "ned", "cov", "msn"]


class MonitoringCallback(DefaultCallbacks):
    def __init__(self,
                 legacy_callbacks_dict: Dict[str, callable] = None,
                 config: Dict = None,
                 logger_config: Dict = None):
        # init default callback
        super().__init__(legacy_callbacks_dict)
        # get callback configuration
        self.logger = logging.getLogger("Callback")
        self.logger.debug("Creating callbacks")
        self.config = config
        if "info_freq" in self.config:
            self.info_freq = self.config["info_freq"]
        else:
            self.info_freq = 20
        if "eval_freq" in self.config:
            self.eval_freq = self.config["eval_freq"]
        else:
            self.eval_freq = 20
        if logger_config is not None:
            self.result_logger = ResultLogger(logger_config, logger_config["logdir"])
        else:
            self.result_logger = None
        self.record_last = "record_last" in self.config and self.config["record_last"]
        self.can_evaluate = self.eval_freq != 0
        self.temp_data = dict()
        self.episode_count = 0
        # init evaluation
        if self.can_evaluate:
            # list of metrics
            if "metrics" in self.config:
                self.metrics = self.config["metrics"]
            else:
                self.metrics = DEFAULT_METRICS
            # structural data
            if ("interaction_data_path" not in self.config) and any(
                    [m in self.metrics for m in ["smq", "icp", "ifn"]]):
                raise ValueError("interaction data is required for measuring smq, icp and ifn")
            elif "interaction_data_path" in self.config:
                interaction_data_path = self.config["interaction_data_path"]
                self.interaction_data = load_data(interaction_data_path, True)
            else:
                self.interaction_data = None
            # semantic data
            if ("semantic_data_path" not in self.config) and \
                    "cmq" in self.metrics:
                raise ValueError("semantic data is required for measuring cmq")
            elif "semantic_data_path" in self.config:
                if "word_count_path" in self.config:
                    # TODO Warning: this is not implemented correctly with the current data_handler class
                    semantic_data_path = self.config["semantic_data_path"]
                    word_count_path = self.config["word_count_path"]
                    word_count = pd.read_csv(word_count_path)
                    self.semantic_data = load_data(semantic_data_path, True)
                    self.semantic_data = self.semantic_data[:, word_count.iloc[0] <= self.semantic_data.shape[0]//2]
                    self.semantic_data = self.semantic_data.dot(self.semantic_data.transpose())
                else:
                    semantic_data_path = self.config["semantic_data_path"]
                    self.semantic_data = load_data(semantic_data_path, True)
            else:
                self.semantic_data = None
            # eval config
            if "exclude_outliers" in self.config:
                self.exclude_outliers = self.config["exclude_outliers"]
            else:
                self.exclude_outliers = True

    def on_algorithm_init(
            self,
            *,
            algorithm: Algorithm,
            **kwargs,
    ) -> None:
        self.logger.debug("starting training {}".format(algorithm.trial_id))

    def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: EpisodeV2,
            **kwargs,
    ) -> None:
        self.episode_count += 1
        self.logger.debug("starting episode {} with id {} (env-idx={})".format(
            self.episode_count, episode.episode_id, episode.env_id))
        self.temp_data = dict()
        self.temp_data["episode_number"] = self.episode_count
        self.temp_data["episode_id"] = episode.episode_id
        self.temp_data["environment_id"] = episode.env_id
        self.temp_data["worker_id"] = worker.worker_index
        self.temp_data["scores"] = []
        self.temp_data["best_state"] = None
        self.temp_data["best_score"] = 0
        if self.can_evaluate:
            for m in self.metrics:
                self.temp_data[m] = []

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: EpisodeV2,
            **kwargs,
    ) -> None:
        env = base_env.get_sub_environments()[0]
        state = env.current_state
        score = episode.last_info_for()["score"] if "score" in episode.last_info_for() else None
        log_text = ""
        if score is not None:
            if self.info_freq > 0 and episode.length % self.info_freq == 0:
                log_text += "ep={} ;; step={} ;; score={:.4f}".format(
                    self.episode_count, episode.length, score
                )
            if self.eval_freq > 0 and episode.length % self.eval_freq == 0:
                results = self.evaluate(state, env)
                for m in results:
                    self.temp_data[m].append(results[m])
                    log_text += " ;; {}={:.4f}".format(m, results[m])
            if log_text != "":
                self.logger.info(log_text)
            if score >= self.temp_data["best_score"]:
                state = env.unflatten(state)
                self.temp_data["best_state"] = state
                self.temp_data["best_score"] = score
            self.temp_data["scores"].append(score)

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: EpisodeV2,
            **kwargs,
    ) -> None:
        if self.record_last:
            # state = episode.last_observation_for()
            env = base_env.get_sub_environments()[0]
            state = env.unflatten(env.current_state)
            self.temp_data["best_state"] = state
        if self.can_evaluate:
            env = base_env.get_sub_environments()[0]
            results = self.evaluate(self.temp_data["best_state"], env)
            for m in results:
                self.temp_data["best_" + m] = results[m]
            self.logger.info("evaluation metrics: " + " , ".join(
                ["{} = {:.4f}".format(m, v) for m, v in results.items()]))
        score = episode.last_info_for()["score"] if "score" in episode.last_info_for() else None
        if score and self.info_freq != 0:
            # self.logger.info("episode {} with id {} (env-inx={}) finished in step {} with the score {:.6f} "
            #             "after applying the action {}".format(
            #     self.episode_count, episode.episode_id, episode.env_id, episode.length,
            #     episode.last_info_for()["score"], episode.last_action_for()
            # ))
            self.logger.info("episode {} with id {} (env-inx={}) finished in step {} with the score {:.6f}".format(
                self.episode_count, episode.episode_id, episode.env_id, episode.length,
                episode.last_info_for()["score"]
            ))
        if self.result_logger is not None:
            self.logger.debug("saving metrics for episode {}".format(self.episode_count))
            self.result_logger.update_on(self.temp_data)

    def evaluate(self, state, env=None):
        assert self.can_evaluate
        results = dict()
        if "n_links" in self.metrics:
            results["n_links"] = state.sum()
        if env is not None:
            state = env.process_state(state)
        features_list = [f for f in [self.interaction_data, self.semantic_data] if f is not None]
        microservices, features_list = process_outliers(state, features_list, self.exclude_outliers)
        if len(np.unique(microservices)) < 1:
            microservices_encoded = np.empty(shape=(1, 0))
        else:
            microservices_encoded = OneHotEncoder().fit_transform(microservices.reshape(-1, 1)).toarray()
        i = 0
        if self.interaction_data is not None:
            interaction_data = features_list[0]
            i += 1
        else:
            interaction_data = None
        if self.semantic_data is not None:
            semantic_data = features_list[i]
        else:
            semantic_data = None
        for m, f in zip(["smq", "icp", "ifn"], [preprocessed_modularity,
                                                preprocessed_inter_call_percentage,
                                                preprocessed_interface_number]):
            if m in self.metrics:
                assert interaction_data is not None
                results[m] = f(microservices_encoded, interaction_data)
        if "cmq" in self.metrics:
            assert semantic_data is not None
            results["cmq"] = preprocessed_modularity(microservices_encoded, semantic_data)
        if "ned" in self.metrics:
            results["ned"] = preprocessed_non_extreme_distribution(microservices)
        if "cov" in self.metrics:
            results["cov"] = coverage(state)
        if "msn" in self.metrics:
            results["msn"] = len(np.unique(microservices))
        return results

    def cleanup(self):
        if self.result_logger is not None:
            self.result_logger.close()


