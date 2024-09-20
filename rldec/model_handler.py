from typing import Optional, Tuple, Any, Dict, List
import logging
import os
import re
import json

from .utils import combine_into
from .config import MODELS_PATH


class ModelHandler:
    """
    Class for handling model related IO tasks such as getting their config and their locations
    """
    # It should be replaced with a more suitable storage method in the future)
    def __init__(self, models_path: Optional[str] = None):
        self.models_path = MODELS_PATH if models_path is None else models_path
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_latest_model(self, app_name: str) -> str:
        search_dir = os.path.join(self.models_path, app_name)
        os.chdir(search_dir)
        paths = filter(os.path.isdir, os.listdir(search_dir))
        paths = [os.path.join(search_dir, f) for f in paths]
        if len(paths) == 0:
            raise FileNotFoundError("Not able to find the checkpoint to the trained model!")
        paths.sort(key=lambda x: os.path.getmtime(x))
        checkpoint_path = paths[-1]
        model_name = os.path.basename(checkpoint_path)
        return model_name

    def get_latest_checkpoint(self, model_path) -> str:
        paths = list(filter(lambda x: os.path.isdir(os.path.join(model_path, x)) and
                                      re.fullmatch(r"checkpoint_\d*", x), os.listdir(model_path)))
        paths.sort(key=lambda x: int(x.replace("checkpoint_", "")))
        checkpoint = paths[-1]
        checkpoint_id = int(checkpoint.replace("checkpoint_", ""))
        checkpoint_path = os.path.join(model_path, checkpoint, f"checkpoint-{checkpoint_id}")
        return checkpoint_path

    def get_model(self, app_name: str, model_name: Optional[str] = None) -> \
            Tuple[str, str]:
        if model_name is None:
            self.logger.info("No model was specified. Searching for the latest model")
            model_name = self.get_latest_model(app_name)
        model_path = os.path.join(self.models_path, app_name, model_name)
        if not os.path.exists(model_path):
            self.logger.debug(f"model path: {model_path}")
            raise FileNotFoundError("Not able to find the specified model!")
        return model_path, model_name

    def list_models(self, app_name: str, language: str = "java") -> List[Dict]:
        models = list()
        for model_name in os.listdir(os.path.join(self.models_path, app_name)):
            model_details = self.get_model_details(app_name, model_name, language)
            if model_details is not None:
                models.append(model_details)
        return models

    def get_model_details(self, app_name: str, model_name: str, language: str = "java", granularity: str = "class") \
            -> Optional[Dict]:
        try:
            model_path, _ = self.get_model(app_name, model_name)
        except FileNotFoundError:
            return
        model_config = self.load_config(model_path)
        model_details = dict()
        model_details["appName"] = app_name
        model_details["language"] = language
        model_details["level"] = granularity
        model_details["experimentID"] = model_name
        model_details["approach"] = model_config["env_config"]["rldec_approach"]
        features = model_config["env_config"]["features_name"]
        model_details["features"] = features if isinstance(features, list) else [features]
        model_details["numIterations"] = model_config["env_config"]["n_iterations"]
        model_details["hyperparamsFile"] = model_config
        raise model_details

    def get_apps(self) -> List[str]:
        return [app for app in os.listdir(self.models_path) if len(self.list_models(app))>0]

    def load_config(self, model_path: str, trainer_generator: Optional[Any] = None) -> Dict:
        with open(os.path.join(model_path, "train_config.json")) as f:
            saved_config = json.load(f)
        if trainer_generator is not None:
            trainer_config = trainer_generator.get_default_config()
            combine_into(saved_config, trainer_config)
        else:
            trainer_config = saved_config
        for c in ["logger_config", "callbacks", "sample_collector"]:
            if c in trainer_config:
                trainer_config.pop(c)
        return trainer_config

    def load_env_config(self, model_path: str) -> Dict:
        with open(os.path.join(model_path, "env_config.json")) as f:
            saved_config = json.load(f)
        return saved_config

