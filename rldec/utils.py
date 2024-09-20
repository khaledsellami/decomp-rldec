import logging
import uuid

import numpy as np
import pandas as pd


def extract_microservices(link_array, n_elements):
    # TODO: optimize this method
    microservices = -1 * np.ones(n_elements)
    offset = 0
    for i in range(n_elements):
        for j in range(0, n_elements - i - 1):
            if link_array[offset + j] == 1:
                first_element_index = i
                second_element_index = j+i+1
                if microservices[first_element_index] == microservices[second_element_index] == -1:
                    new_microservice = microservices.max() + 1
                    microservices[first_element_index] = new_microservice
                    microservices[second_element_index] = new_microservice
                elif microservices[first_element_index] == -1:
                    microservices[first_element_index] = microservices[second_element_index]
                elif microservices[second_element_index] == -1:
                    microservices[second_element_index] = microservices[first_element_index]
                else:
                    min_micro = np.min([microservices[first_element_index], microservices[second_element_index]])
                    max_micro = np.max([microservices[first_element_index], microservices[second_element_index]])
                    microservices[microservices == max_micro] = min_micro
        offset += (n_elements - i - 1)
    return microservices


def generate_id(size: int = 8):
    logger = logging.getLogger('Basic')
    if size > 32:
        logger.warning("size is larger than 32: generating unique id with size 32 instead!")
    return str(uuid.uuid4()).replace("-", "")[:min(32, size)]


def combine_into(d: dict, combined: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            if k in combined and combined[k] is None:
                combined[k] = {}
                combine_into(v, combined[k])
            else:
                combine_into(v, combined.setdefault(k, {}))
        else:
            combined[k] = v


def load_data(path, as_numpy: bool = False) -> pd.DataFrame|np.ndarray:
    if path.endswith("parquet") or path.endswith("pq"):
        data = pd.read_parquet(path)
    elif path.endswith("csv"):
        data = pd.read_csv(path, index_col=0)
    elif path.endswith("npy"):
        data = np.load(path)
    else:
        raise ValueError("Unrecognized data_format {}!".format(format))
    if as_numpy and isinstance(data, pd.DataFrame):
        return data.values
    return data
