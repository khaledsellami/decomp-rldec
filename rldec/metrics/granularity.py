import numpy as np

from .utils import process_outliers


def non_extreme_distribution(microservices: np.ndarray, s_min: int = 5, s_max: int = 19,
                             exclude_outliers: bool = True) -> float:
    """
    Process outliers and calculate the Non-Extreme Distribution of the given decomposition.
    :param microservices: microservices index for each class/method (List[int])
    :param s_min: minimum threshold
    :param s_max: maximum threshold
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: Non-Extreme Distribution value
    """
    microservices, _ = process_outliers(microservices, [], exclude_outliers)
    return preprocessed_non_extreme_distribution(microservices, s_min, s_max)


def preprocessed_non_extreme_distribution(microservices: np.ndarray, s_min: int = 5, s_max: int = 19,
                                          method: str = "minmax", epsilon: float = 0.5, K: int = 5) -> float:
    """
    Calculate the Non-Extreme Distribution of the given decomposition.
    :param microservices: microservices index for each class/method (List[int])
    :param s_min: minimum threshold (only needed for "minmax" method)
    :param s_max: maximum threshold (only needed for "minmax" method)
    :param method: NED calculation method.
    :param epsilon: controls the bandwidth for acceptable size (only needed for "avg" method)
    :param K: number of target microservices (only needed for "avg" method)
    :return: Non-Extreme Distribution value
    """
    unique, counts = np.unique(microservices, return_counts=True)
    n_microservices = len(unique)
    n_classes = len(microservices)
    if n_microservices < 1:
        return 1
    if method == "minmax":
        non_extreme = ((counts >= s_min)*(counts <= s_max)).sum()
        ned = 1 - non_extreme / n_microservices
    elif method == "avg":
        non_extreme = ((counts >= ((1-epsilon)*n_classes/K))*(
                counts <= ((1+epsilon)*n_classes/K))*counts).sum()
        ned = 1 - non_extreme / n_classes
    else:
        raise ValueError("Unidentified option '{}' for parameter 'method'!".format(method))
    return ned


def coverage(microservices: np.ndarray) -> float:
    """
    Calculate the class/method coverage in the decomposition
    :param microservices: microservices index for each class/method (List[int]) where outliers are referred by -1
    :return: coverage value
    """
    return (microservices != -1).sum() / len(microservices)


def microservices_number(microservices: np.ndarray, exclude_outliers: bool = True) -> int:
    """
    Calculate the number of microservices.
    :param microservices: microservices index for each class/method (List[int])
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: number of microservices in the decomposition
    """
    n_micro = len(np.unique(microservices)) - np.any(microservices == -1)
    if not exclude_outliers:
        n_micro += np.sum(microservices == -1)
    return n_micro