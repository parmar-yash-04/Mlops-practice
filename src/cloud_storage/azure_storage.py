import os
import sys
from src.configuration.azure_connection import AzureBlobClient
from src.constants import AZURE_CONTAINER_NAME, MODEL_PUSHER_BLOB_KEY
from src.exception import MyException
from src.logger import logging


class AzureCloudStorage:
    def __init__(self):
        self.azure_client = AzureBlobClient()

    def upload_model(self, local_file_path: str, container_name: str = AZURE_CONTAINER_NAME, blob_key: str = MODEL_PUSHER_BLOB_KEY):
        try:
            blob_client = self.azure_client.get_blob_client(container_name, blob_key)
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            logging.info(f"Model uploaded to Azure Blob: {container_name}/{blob_key}")
        except Exception as e:
            raise MyException(e, sys)

    def download_model(self, download_path: str, container_name: str = AZURE_CONTAINER_NAME, blob_key: str = MODEL_PUSHER_BLOB_KEY):
        try:
            blob_client = self.azure_client.get_blob_client(container_name, blob_key)
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            with open(download_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            logging.info(f"Model downloaded from Azure Blob: {container_name}/{blob_key}")
        except Exception as e:
            raise MyException(e, sys)

    def model_exists(self, container_name: str = AZURE_CONTAINER_NAME, blob_key: str = MODEL_PUSHER_BLOB_KEY) -> bool:
        try:
            blob_client = self.azure_client.get_blob_client(container_name, blob_key)
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False
