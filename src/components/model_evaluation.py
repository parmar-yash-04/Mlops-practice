import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from src.constants import *
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from src.utils.main_utils import load_object, load_numpy_array
from src.entity.local_estimator import LocalEstimator


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Evaluation")

            trained_model_path = self.model_trainer_artifact.trained_model_file_path
            new_model_accuracy = self.model_trainer_artifact.test_accuracy
            logging.info(f"New model accuracy: {new_model_accuracy:.4f}")

            estimator = LocalEstimator()
            model_exists = estimator.is_model_available()

            if not model_exists:
                logging.info("No existing model found in local registry. Accepting new model.")
                return ModelEvaluationArtifact(
                    model_accepted=True,
                    changed_accuracy=0.0,
                    trained_model_path=trained_model_path,
                    test_accuracy=new_model_accuracy,
                )

            logging.info("Existing model found in local registry. Evaluating...")
            temp_dir = os.path.join(ARTIFACT_DIR, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            existing_model_path = os.path.join(temp_dir, "existing_model.pkl")
            estimator.pull_model(existing_model_path)
            existing_model = load_object(existing_model_path)

            test_arr = load_numpy_array(
                self.data_transformation_artifact.transformed_test_file_path
            )
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            y_pred_existing = existing_model.predict(x_test)
            existing_model_accuracy = accuracy_score(y_test, y_pred_existing)
            logging.info(f"Existing model accuracy: {existing_model_accuracy:.4f}")

            changed_accuracy = new_model_accuracy - existing_model_accuracy
            logging.info(f"Accuracy change: {changed_accuracy:.4f}")

            threshold = self.model_evaluation_config.changed_threshold_score
            if changed_accuracy >= threshold:
                logging.info(
                    f"Accuracy improved by {changed_accuracy:.4f} (threshold: {threshold}). Accepting model."
                )
                model_accepted = True
            else:
                logging.info(
                    f"Accuracy change {changed_accuracy:.4f} below threshold {threshold}. Rejecting model."
                )
                model_accepted = False

            return ModelEvaluationArtifact(
                model_accepted=model_accepted,
                changed_accuracy=changed_accuracy,
                trained_model_path=trained_model_path,
                test_accuracy=new_model_accuracy,
            )

        except Exception as e:
            raise MyException(e, sys)
