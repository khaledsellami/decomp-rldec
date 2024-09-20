import json
import os
from typing import List, Tuple, Optional

import numpy as np

from .metrics import vectorized_modularity
from .utils import load_data
from .config import SUPPORTED_FORMATS


class Features:
    def __init__(self, name: str, features_path: str, atoms: List[str], supp_atoms_path: Optional[str] = None, both_axes: bool = True,
                 data_format: str = "npy"):
        """
        :param name: name of the features (example: structural, dynamic, sematic, etc)
        :param features_path: path to the features matrix
        :param atoms_path: path to the names of the atoms used in rldec
        :param supp_atoms_path: path to the names of the atoms that correspond to the features_matrix
        :param both_axes: to filter out atoms from both axes or not
        :param data_format: format of the features matrix
        """
        self.name = name
        self.data_format = data_format
        if self.data_format == "npy":
            with open(supp_atoms_path, "r") as f:
                supp_atoms = json.load(f)
            features = np.load(features_path)
        else:
            features = load_data(features_path)
            supp_atoms = features.index.to_list()
            features = features.values
        assert len(supp_atoms) == features.shape[0]
        # get list of atoms that have features and are used by rldec
        supp_atoms_map = {i: a for i, a in enumerate(supp_atoms) if a in atoms}
        assert len(supp_atoms_map) > 0
        self.atoms = list(supp_atoms_map.values())
        to_include = list(supp_atoms_map.keys())
        # include only the features of the considered atoms
        if both_axes:
            self.features = features[to_include][:, to_include]
        else:
            self.features = features[to_include]
        # create list that maps the indices of the considered atoms to the rldec atoms
        self.atoms_map = [atoms.index(a) for a in self.atoms]
        if both_axes:
            assert self.features.shape[0] == len(self.atoms_map) == self.features.shape[1]
        else:
            assert self.features.shape[0] == len(self.atoms_map)

    def vectorized_modularity(self, microservices: np.ndarray, exclude_outliers: bool = True) -> float:
        return vectorized_modularity(microservices[self.atoms_map], self.features, exclude_outliers)

    def get_elements(self):
        return len(self.atoms)


def get_feature_paths(name: str, data_path: str, features_name: str = None, granularity: str = "class",
                      data_format: str = "npy") -> Tuple[str, str]:
    if data_format == "npy":
        # Warning: this is the old way of loading features. It should be depricated in the future
        if name == "structural":
            features_name = f"{granularity}_interactions.npy" if features_name is None else features_name
            features_path = os.path.join(data_path, "structural_data", features_name)
            supp_atoms_path = os.path.join(data_path, "structural_data", f"{granularity}_names.json")
        elif name == "semantic":
            features_name = f"{granularity}_word_count.npy" if features_name is None else features_name
            features_path = os.path.join(data_path, "semantic_data", features_name)
            supp_atoms_path = os.path.join(data_path, "semantic_data", f"{granularity}_names.json")
        elif name == "dynamic":
            features_name = f"{granularity}_calls.npy" if features_name is None else features_name
            features_path = os.path.join(data_path, "dynamic_analysis", features_name)
            supp_atoms_path = os.path.join(data_path, "dynamic_analysis", f"{granularity}_names.json")
        else:
            raise NotImplementedError("The feature name {} is not supported yet".format(name))
    else:
        if name == "structural":
            features_name = f"{granularity}_interactions.{data_format}" if features_name is None else features_name
            features_path = os.path.join(data_path, features_name)
            supp_atoms_path = None
        elif name == "semantic":
            features_name = f"{granularity}_word_count.{data_format}" if features_name is None else features_name
            features_path = os.path.join(data_path, features_name)
            supp_atoms_path = None
        elif name == "dynamic":
            features_name = f"{granularity}_dynamic_calls.{data_format}" if features_name is None else features_name
            features_path = os.path.join(data_path, features_name)
            supp_atoms_path = None
        else:
            raise NotImplementedError("The feature name {} is not supported yet".format(name))
    return features_path, supp_atoms_path


def load_features(name: str, data_path: str, atoms: List[str], features_name: str = None, granularity: str = "class",
                  data_format: str = "npy") -> Features:
    if not data_format in SUPPORTED_FORMATS:
        raise ValueError(f"Data format {data_format} is not supported")
    features_path, supp_atoms_path = get_feature_paths(name, data_path, features_name, granularity, data_format)
    features = Features(name, features_path, atoms, supp_atoms_path, both_axes=True, data_format=data_format)
    return features
