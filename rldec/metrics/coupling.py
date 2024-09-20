import numpy as np

from .utils import process_outliers, one_hot_encode


def inter_call_percentage(microservices: np.ndarray, interactions: np.ndarray,
                          exclude_outliers: bool = True) -> float:
    """
    One hot encode the input microservice array and measure the inter-call percentage.
    :param microservices: microservices index for each class/method (List[int])
    :param interactions: class/method interaction matrix
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: inter-call percentage value
    """
    microservices, f = process_outliers(microservices, [interactions], exclude_outliers)
    interactions = f[0]
    n_microservices = len(np.unique(microservices))
    if n_microservices < 1:
        return 1
    microservices = one_hot_encode(microservices)
    return preprocessed_inter_call_percentage(microservices, interactions)


def preprocessed_inter_call_percentage(microservices: np.ndarray, interactions: np.ndarray) -> float:
    """
    Calculate the inter-call percentage.
    :param microservices: microservices index for each class/method (one-hot encoded)
    :param interactions: class/method interaction matrix
    :return: inter-call percentage value
    """
    n_microservices = microservices.shape[1]
    if n_microservices < 1:
        return 1
    interactions = interactions + 1
    total = np.log(interactions)
    total = microservices.transpose().dot(total).dot(microservices)
    inter = total.sum() - total.diagonal().sum()
    total = total.sum()
    if total == 0:
        return 0
    else:
        return inter/total


def interface_number(microservices: np.ndarray, interactions: np.ndarray,
                     exclude_outliers: bool = True) -> float:
    """
    One hot encode the input microservice array and measure the number of interface classes/methods.
    :param microservices: microservices index for each class/method (List[int])
    :param interactions: class/method interaction matrix
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: Interface number
    """
    microservices, f = process_outliers(microservices, [interactions], exclude_outliers)
    interactions = f[0]
    n_microservices = len(np.unique(microservices))
    if n_microservices < 1:
        return interactions.shape[0]
    microservices = one_hot_encode(microservices)
    return preprocessed_interface_number(microservices, interactions)


def preprocessed_interface_number(microservices: np.ndarray, interactions: np.ndarray) -> float:
    """
    Calculate the number of interface classes/methods.
    :param microservices: microservices index for each class/method (one-hot encoded)
    :param interactions: class/method interaction matrix
    :return: Interface number
    """
    n_microservices = microservices.shape[1]
    if n_microservices <= 1:
        return 0
    interfaces = microservices.transpose().dot(interactions)
    interfaces = interfaces * (1 - microservices.transpose())
    return (interfaces.sum(0) > 0).sum() / n_microservices
