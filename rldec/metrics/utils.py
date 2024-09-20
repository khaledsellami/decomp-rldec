import re
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import OneHotEncoder


METRICS_RANGES = dict(
    chm=(0, 1),
    chd=(0, 1),
    smq=(-1, 1),
    cmq=(-1, 1),
    ned=(0, 1),
    icp=(0, 1),
    cov=(0, 1)
)


def generate_random_decomposition(atom_names: List[str], max_micro: int = 20):
    decomposition = np.random.randint(max_micro, size=len(atom_names))
    return decomposition


def tokenize_name(text: str) -> List[str]:
    return [i for w in text.split(" ") for x in w.split("_")
            for i in re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', x).split()]


def process_outliers(decomposition: np.ndarray,
                     features_list: List[np.ndarray] = None,
                     exclude_outliers: bool = True,
                     both_axes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Excludes outlier classes/methods from list of microservices and related features and increments the index otherwise.
    :param decomposition: microservices index for each class/method (List[int])
    :param features_list: features that are used for evaluation
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :param both_axes: True if features where both axes represent the classes/methods
    :return: processed microservices array and features array
    """
    if exclude_outliers:
        include = decomposition != -1
        if features_list is not None and len(features_list) > 0:
            new_features_list = list()
            for features in features_list:
                if both_axes and features.shape[0] == features.shape[1]:
                    new_features_list.append(features[include][:, include])
                else:
                    new_features_list.append(features[include])
        else:
            new_features_list = features_list
        new_decomposition = decomposition[include]
    else:
        new_decomposition = decomposition.copy()
        x = new_decomposition == -1
        new_features_list = features_list
        new_decomposition[x] = new_decomposition.max() + 1 + np.arange(x.sum())
    return new_decomposition, new_features_list


def one_hot_encode(microservices: np.ndarray) -> np.ndarray:
    microservices = OneHotEncoder().fit_transform(microservices.reshape(-1, 1)).toarray()
    return microservices
