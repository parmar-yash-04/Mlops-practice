import os
import sys
from azure.storage.blob import BlobServiceClient
from src.constants import AZURE_STORAGE_CONNECTION_STRING_KEY
from src.exception import MyException


class AzureBlobClient:
    def __init__(self):
        try:
            conn_str = os.getenv(AZURE_STORAGE_CONNECTION_STRING_KEY)
            if conn_str is None:
                raise Exception(
                    f"Environment variable '{AZURE_STORAGE_CONNECTION_STRING_KEY}' is not set"
                )
            self.blob_service = BlobServiceClient.from_connection_string(conn_str)
        except Exception as e:
            raise MyException(e, sys)

    def get_container_client(self, container_name: str):
        try:
            return self.blob_service.get_container_client(container_name)
        except Exception as e:
            raise MyException(e, sys)

    def get_blob_client(self, container_name: str, blob_key: str):
        try:
            return self.blob_service.get_blob_client(
                container=container_name, blob=blob_key
            )
        except Exception as e:
            raise MyException(e, sys)
