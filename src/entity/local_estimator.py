import os
import sys
import shutil
from src.constants import LOCAL_MODEL_REGISTRY_PATH
from src.exception import MyException
from src.logger import logging


class LocalEstimator:
    def __init__(self):
        self.registry_path = LOCAL_MODEL_REGISTRY_PATH
        os.makedirs(self.registry_path, exist_ok=True)

    def push_model(self, model_path: str):
        try:
            dest = os.path.join(self.registry_path, "model.pkl")
            shutil.copy2(model_path, dest)
            logging.info(f"Model saved to local registry: {dest}")
        except Exception as e:
            raise MyException(e, sys)

    def pull_model(self, download_path: str):
        try:
            src = os.path.join(self.registry_path, "model.pkl")
            shutil.copy2(src, download_path)
            logging.info(f"Model pulled from local registry to: {download_path}")
        except Exception as e:
            raise MyException(e, sys)

    def is_model_available(self) -> bool:
        return os.path.exists(os.path.join(self.registry_path, "model.pkl"))
