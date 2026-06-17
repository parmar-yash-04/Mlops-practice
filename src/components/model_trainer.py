import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.constants import *
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.utils.main_utils import load_numpy_array, save_object, read_yaml


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def train_model(self, x_train, y_train):
        try:
            model_config = read_yaml(self.model_trainer_config.model_config_file_path)
            logging.info(f"Model config: {model_config}")

            rf = RandomForestClassifier(
                n_estimators=model_config.get("n_estimators", MODEL_TRAINER_N_ESTIMATORS),
                min_samples_split=model_config.get("min_samples_split", MODEL_TRAINER_MIN_SAMPLES_SPLIT),
                min_samples_leaf=model_config.get("min_samples_leaf", MODEL_TRAINER_MIN_SAMPLES_LEAF),
                max_depth=model_config.get("max_depth", None),
                criterion=model_config.get("criterion", "entropy"),
                random_state=model_config.get("random_state", 101),
            )
            rf.fit(x_train, y_train)
            logging.info("RandomForest model trained successfully")
            return rf
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Trainer")

            train_arr = load_numpy_array(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array(self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Loaded train array: {train_arr.shape}")
            logging.info(f"Loaded test array: {test_arr.shape}")

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info(f"Train features: {x_train.shape}, Train target: {y_train.shape}")
            logging.info(f"Test features: {x_test.shape}, Test target: {y_test.shape}")

            model = self.train_model(x_train, y_train)

            y_train_pred = model.predict(x_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            logging.info(f"Train accuracy: {train_accuracy:.4f}")

            y_test_pred = model.predict(x_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            logging.info(f"Test accuracy: {test_accuracy:.4f}")

            model_accuracy = test_accuracy

            if model_accuracy < self.model_trainer_config.expected_score:
                logging.warning(
                    f"Model accuracy {model_accuracy:.4f} is below expected threshold "
                    f"{self.model_trainer_config.expected_score}"
                )

            trained_model_dir = self.model_trainer_config.trained_model_dir
            trained_model_path = self.model_trainer_config.trained_model_path
            os.makedirs(trained_model_dir, exist_ok=True)
            save_object(trained_model_path, model)
            logging.info(f"Model saved: {trained_model_path}")

            logging.info("Model Trainer completed successfully")

            return ModelTrainerArtifact(
                trained_model_file_path=trained_model_path,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                model_accuracy=model_accuracy,
            )

        except Exception as e:
            raise MyException(e, sys)
