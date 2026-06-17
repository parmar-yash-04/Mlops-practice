import os
import sys
from src.constants import *
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from src.entity.azure_estimator import AzureEstimator


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ):
        self.model_pusher_config = model_pusher_config
        self.model_evaluation_artifact = model_evaluation_artifact

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Pusher")

            model_accepted = self.model_evaluation_artifact.model_accepted

            if not model_accepted:
                logging.info("Model was rejected. Skipping push to Azure Blob.")
                return ModelPusherArtifact(
                    model_pushed=False,
                    blob_path="",
                )

            logging.info("Model accepted. Pushing to Azure Blob...")
            azure_estimator = AzureEstimator()

            trained_model_path = self.model_evaluation_artifact.trained_model_path
            azure_estimator.push_model(trained_model_path)

            preprocessing_obj_path = self.model_pusher_config.preprocessing_obj_path
            if os.path.exists(preprocessing_obj_path):
                preprocessing_blob_key = "preprocessing.pkl"
                from src.cloud_storage.azure_storage import AzureCloudStorage
                cloud_storage = AzureCloudStorage()
                cloud_storage.upload_model(
                    local_file_path=preprocessing_obj_path,
                    container_name=AZURE_CONTAINER_NAME,
                    blob_key=preprocessing_blob_key,
                )
                logging.info(f"Preprocessing object uploaded to Azure Blob: {preprocessing_blob_key}")

            blob_path = f"{AZURE_CONTAINER_NAME}/{MODEL_PUSHER_BLOB_KEY}"
            logging.info(f"Model pusher completed. Model pushed to: {blob_path}")

            return ModelPusherArtifact(
                model_pushed=True,
                blob_path=blob_path,
            )

        except Exception as e:
            raise MyException(e, sys)
