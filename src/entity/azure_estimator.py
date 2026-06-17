import os
import sys
from src.cloud_storage.azure_storage import AzureCloudStorage
from src.constants import AZURE_CONTAINER_NAME, MODEL_PUSHER_BLOB_KEY
from src.exception import MyException
from src.logger import logging


class AzureEstimator:
    def __init__(self):
        self.cloud_storage = AzureCloudStorage()
        self.container_name = AZURE_CONTAINER_NAME
        self.blob_key = MODEL_PUSHER_BLOB_KEY

    def push_model(self, model_path: str):
        try:
            logging.info(f"Pushing model to Azure Blob: {model_path}")
            self.cloud_storage.upload_model(
                local_file_path=model_path,
                container_name=self.container_name,
                blob_key=self.blob_key,
            )
            logging.info("Model pushed to Azure Blob successfully")
        except Exception as e:
            raise MyException(e, sys)

    def pull_model(self, download_path: str):
        try:
            logging.info(f"Pulling model from Azure Blob to: {download_path}")
            self.cloud_storage.download_model(
                download_path=download_path,
                container_name=self.container_name,
                blob_key=self.blob_key,
            )
            logging.info("Model pulled from Azure Blob successfully")
        except Exception as e:
            raise MyException(e, sys)

    def is_model_available(self) -> bool:
        return self.cloud_storage.model_exists(
            container_name=self.container_name,
            blob_key=self.blob_key,
        )
