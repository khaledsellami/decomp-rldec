import numpy as np

from .utils import process_outliers, one_hot_encode


def vectorized_modularity(microservices: np.ndarray,
                          features: np.ndarray,
                          exclude_outliers: bool = True) -> float:
    """
    One hot encodes the input microservice array and measures the modularity.
    :param microservices: microservices index for each class/method (List[int])
    :param features: features that are used for evaluation
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: modularity value
    """
    # TODO: Optimize further
    microservices, f = process_outliers(microservices, [features], exclude_outliers)
    features = f[0]
    n_microservices = len(np.unique(microservices))
    if n_microservices < 2:
        return 0
    microservices = one_hot_encode(microservices)
    return preprocessed_modularity(microservices, features)


def preprocessed_modularity(microservices: np.ndarray, features: np.ndarray) -> float:
    """
    Calculate the modular quality for the given decomposition and features.
    :param microservices: microservices index for each class/method (one-hot encoded)
    :param features: features that are used for evaluation
    :return: modularity value
    """
    n_microservices = microservices.shape[1]
    if n_microservices < 2:
        return 0
    features = features > 0
    element_number = microservices.sum(axis=0).reshape(-1, 1)
    element_number = element_number.dot(element_number.transpose())
    mod_matrix = microservices.transpose().dot(features).dot(microservices) / element_number
    cohesion = mod_matrix.diagonal().sum()
    coupling = mod_matrix.sum() - cohesion
    return cohesion / n_microservices - coupling / (n_microservices * (n_microservices - 1))