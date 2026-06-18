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
from src.entity.local_estimator import LocalEstimator


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
                logging.info("Model was rejected. Skipping push to local registry.")
                return ModelPusherArtifact(
                    model_pushed=False,
                    model_registry_path="",
                )

            logging.info("Model accepted. Pushing to local registry...")
            estimator = LocalEstimator()

            trained_model_path = self.model_evaluation_artifact.trained_model_path
            estimator.push_model(trained_model_path)

            preprocessing_obj_path = self.model_pusher_config.preprocessing_obj_path
            if os.path.exists(preprocessing_obj_path):
                import shutil
                preprocessing_registry_path = os.path.join(LOCAL_MODEL_REGISTRY_PATH, "preprocessing.pkl")
                shutil.copy2(preprocessing_obj_path, preprocessing_registry_path)
                logging.info(f"Preprocessing object saved to local registry: {preprocessing_registry_path}")

            registry_path = str(os.path.join(LOCAL_MODEL_REGISTRY_PATH, "model.pkl"))
            logging.info(f"Model pusher completed. Model pushed to: {registry_path}")

            return ModelPusherArtifact(
                model_pushed=True,
                model_registry_path=registry_path,
            )

        except Exception as e:
            raise MyException(e, sys)
