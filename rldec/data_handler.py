import json
import logging
import os
from typing import List, Any

import numpy as np
import pandas as pd

from .features import get_feature_paths
from .clients.parsingClient import ParsingClient
from .utils import load_data
from .config import SUPPORTED_FORMATS


class DataHandler:
    DEFAULT_DATA_PATH = os.path.join(os.curdir, "data")

    def __init__(self, app_name: str, data_path: str = None, app_repo: str = None, granularity: str = "class",
                 is_distributed: bool = False, data_format: str = "parquet", *args, **kwargs):
        assert granularity in ["class", "method"]
        if not data_format in SUPPORTED_FORMATS:
            raise ValueError(f"Data format {data_format} is not supported")
        if data_path is None:
            data_path = self.DEFAULT_DATA_PATH
        self.app_name = app_name
        self.app_repo = app_repo
        self.granularity = granularity
        self.is_distributed = is_distributed
        self.app_path = os.path.join(data_path, app_name)
        self.data_format = data_format
        self.logger = logging.getLogger("Data_handler")
        self.parsing_client = ParsingClient(self.app_name, self.app_repo, "java", granularity, is_distributed,
                                            *args, **kwargs)

    def get_names(self) -> List[str]:
        if self.data_format == "npy":
            filename = f"{self.granularity}_names.json"
            path = os.path.join(self.app_path, "semantic_data", filename)
            if not os.path.exists(path):
                self.logger.debug("data not found in local directory")
                self.pull(filename, path)
            with open(path, "r") as f:
                names = json.load(f)
        else:
            filename = f"{self.granularity}_tfidf.{self.data_format}"
            path = os.path.join(self.app_path, filename)
            if not os.path.exists(path):
                self.logger.debug("data not found in local directory")
                self.pull(filename, path)
            names = load_data(path).index.to_list()
        return names

    def get_path(self, file: str):
        # TODO improve to handle all types of files in the future
        if self.data_format == "npy":
            assert any([file==f.format(self.granularity) for f in ["{}_names.json", "{}_tfidf.npy", "{}_vocabulary.csv",
                                                                   "{}_interactions.npy", "{}_word_counts.npy"]])
            if any([file==f.format(self.granularity) for f in ["{}_names.json", "{}_tfidf.npy", "{}_vocabulary.csv",
                                                               "{}_word_counts.npy"]]):
                path = get_feature_paths("semantic", self.app_path, file)[0]
            else:
                path = get_feature_paths("structural", self.app_path, file)[0]
        else:
            path = os.path.join(self.app_path, file)
        if not os.path.exists(path):
            self.logger.debug("data not found in local directory")
            self.pull(file, path)
        return path

    def pull_all(self):
        if self.data_format == "npy":
            for file in ["{}_names.json", "{}_tfidf.npy", "{}_word_count.npy"]:
                path = get_feature_paths("semantic", self.app_path, file.format(self.granularity))[0]
                if not os.path.exists(path):
                    self.pull(file, path)
            for file in ["{}_interactions.npy"]:
                path = get_feature_paths("structural", self.app_path, file.format(self.granularity))[0]
                if not os.path.exists(path):
                    self.pull(file.format(self.granularity), path)
        else:
            for file in ["{}_tfidf.{}", "{}_word_count.{}", "{}_interactions.{}"]:
                path = os.path.join(self.app_path, file.format(self.granularity, self.data_format))
                if not os.path.exists(path):
                    self.logger.debug(f"data not found in {path}")
                    self.pull(file.format(self.granularity, self.data_format), path)

    def pull(self, file: str, save_path: str):
        self.logger.debug(f"pulling {file} from the parsing service")
        if file.endswith("names.json"):
            data = [c for c in self.parsing_client.get_names()]
            names = data
        elif file.endswith(f"tfidf.{self.data_format}"):
            data = self.parsing_client.get_tfidf()
            names = list(data.index.values)
        elif file.endswith(f"word_count.{self.data_format}"):
            data = self.parsing_client.get_word_counts()
            names = list(data.index.values)
        elif file.endswith(f"interactions.{self.data_format}"):
            data = self.parsing_client.get_interactions()
            names = list(data.index.values)
        elif file.endswith(f"calls.{self.data_format}"):
            data = self.parsing_client.get_calls()
            names = list(data.index.values)
        else:
            raise NotImplementedError(f"Unknown file type {file}")
        if self.data_format == "npy":
            data = data.values
            if file.endswith(f"tfidf.{self.data_format}"):
                self.save(names, os.path.join(os.path.dirname(save_path), f"{self.granularity}_names.json"))
        self.save(data, save_path)

    def save(self, data: Any, save_path: str, override: bool = False):
        if os.path.exists(save_path) and not override:
            self.logger.warning(f"file {save_path} already exists!")
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".json"):
            with open(save_path, "w") as f:
                json.dump(data, f)
        elif save_path.endswith(".npy"):
            np.save(save_path, data)
        elif save_path.endswith(".csv"):
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            data.to_csv(save_path)
        elif save_path.endswith(".parquet"):
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            data.to_parquet(save_path)
        else:
            raise ValueError(f"Unsupported file format {save_path.split('.')[-1]}")