import os
import sys
import yaml
import pickle
import numpy as np
from src.exception import MyException
from src.constants import SCHEMA_FILE_PATH


def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise MyException(e, sys)


def write_yaml(file_path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f)
    except Exception as e:
        raise MyException(e, sys)


def save_object(file_path: str, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise MyException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise MyException(e, sys)


def save_numpy_array(file_path: str, arr: np.ndarray):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, arr)
    except Exception as e:
        raise MyException(e, sys)


def load_numpy_array(file_path: str) -> np.ndarray:
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise MyException(e, sys)


def load_schema() -> dict:
    return read_yaml(SCHEMA_FILE_PATH)
