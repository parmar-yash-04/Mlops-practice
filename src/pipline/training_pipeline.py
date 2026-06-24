import os
import sys
from datetime import datetime
import mlflow
from src.exception import MyException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from src.constants import *


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig(
            root_dir=ARTIFACT_DIR,
            feature_store_path=os.path.join(
                ARTIFACT_DIR,
                DATA_INGESTION_DIR_NAME,
                DATA_INGESTION_FEATURE_STORE_DIR,
                FILE_NAME
            ),
            ingested_dir=os.path.join(
                ARTIFACT_DIR,
                DATA_INGESTION_DIR_NAME,
                DATA_INGESTION_INGESTED_DIR
            ),
            train_path=os.path.join(
                ARTIFACT_DIR,
                DATA_INGESTION_DIR_NAME,
                DATA_INGESTION_INGESTED_DIR,
                TRAIN_FILE_NAME
            ),
            test_path=os.path.join(
                ARTIFACT_DIR,
                DATA_INGESTION_DIR_NAME,
                DATA_INGESTION_INGESTED_DIR,
                TEST_FILE_NAME
            ),
            test_size=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.data_validation_config = DataValidationConfig(
            root_dir=ARTIFACT_DIR,
            report_file_path=os.path.join(
                ARTIFACT_DIR, 
                DATA_VALIDATION_DIR_NAME,
                DATA_VALIDATION_REPORT_FILE_NAME
            ),
        )
        self.data_transformation_config = DataTransformationConfig(
            root_dir=ARTIFACT_DIR,
            transformed_train_dir=os.path.join(
                ARTIFACT_DIR,
                DATA_TRANSFORMATION_DIR_NAME,
                DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
            ),
            transformed_test_dir=os.path.join(
                ARTIFACT_DIR,
                DATA_TRANSFORMATION_DIR_NAME,
                DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
            ),
            preprocessing_obj_path=os.path.join(
                ARTIFACT_DIR,
                DATA_TRANSFORMATION_DIR_NAME,
                DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                PREPROCSSING_OBJECT_FILE_NAME
            ),
        )
        self.model_trainer_config = ModelTrainerConfig(
            root_dir=ARTIFACT_DIR,
            trained_model_dir=os.path.join(
                ARTIFACT_DIR,
                MODEL_TRAINER_DIR_NAME,
                MODEL_TRAINER_TRAINED_MODEL_DIR
            ),
            trained_model_path=os.path.join(
                ARTIFACT_DIR,
                MODEL_TRAINER_DIR_NAME,
                MODEL_TRAINER_TRAINED_MODEL_DIR,
                MODEL_TRAINER_TRAINED_MODEL_NAME
            ),
            expected_score=MODEL_TRAINER_EXPECTED_SCORE,
            model_config_file_path=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH,
        )
        self.model_evaluation_config = ModelEvaluationConfig(
            root_dir=ARTIFACT_DIR,
            test_data_path=os.path.join(
                ARTIFACT_DIR,
                DATA_TRANSFORMATION_DIR_NAME,
                DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                TEST_FILE_NAME.replace(".csv", ".npy"),
            ),
            trained_model_path=os.path.join(
                ARTIFACT_DIR,
                MODEL_TRAINER_DIR_NAME,
                MODEL_TRAINER_TRAINED_MODEL_DIR,
                MODEL_TRAINER_TRAINED_MODEL_NAME,
            ),
            changed_threshold_score=MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE,
        )
        self.model_pusher_config = ModelPusherConfig(
            root_dir=ARTIFACT_DIR,
            trained_model_path=os.path.join(
                ARTIFACT_DIR,
                MODEL_TRAINER_DIR_NAME,
                MODEL_TRAINER_TRAINED_MODEL_DIR,
                MODEL_TRAINER_TRAINED_MODEL_NAME,
            ),
            preprocessing_obj_path=os.path.join(
                ARTIFACT_DIR,
                DATA_TRANSFORMATION_DIR_NAME,
                DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                PREPROCSSING_OBJECT_FILE_NAME,
            ),
        )

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed: Train={artifact.train_file_path}, Test={artifact.test_file_path}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Data Validation")
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )
            artifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed: Status={artifact.validation_status}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_transformation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Data Transformation")
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )
            artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed: Train={artifact.transformed_train_file_path}, Test={artifact.transformed_test_file_path}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Trainer")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )
            artifact = model_trainer.initiate_model_trainer()
            logging.info(
                f"Model Trainer completed: Accuracy={artifact.model_accuracy:.4f}, "
                f"Model={artifact.trained_model_file_path}"
            )
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Evaluation")
            from src.components.model_evaluation import ModelEvaluation
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            artifact = model_evaluation.initiate_model_evaluation()
            logging.info(
                f"Model Evaluation completed: Accepted={artifact.model_accepted}, "
                f"Accuracy={artifact.test_accuracy:.4f}"
            )
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_pusher(
        self, model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Pusher")
            from src.components.model_pusher import ModelPusher
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_evaluation_artifact=model_evaluation_artifact,
            )
            artifact = model_pusher.initiate_model_pusher()
            logging.info(f"Model Pusher completed: Pushed={artifact.model_pushed}")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            run_name = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "pipeline_type": "training",
                    "model_config_file": MODEL_TRAINER_MODEL_CONFIG_FILE_PATH,
                })

                data_ingestion_artifact = self.start_data_ingestion()

                if hasattr(data_ingestion_artifact, "train_file_path"):
                    mlflow.log_params({
                        "train_data": str(data_ingestion_artifact.train_file_path),
                        "test_data": str(data_ingestion_artifact.test_file_path),
                    })

                self.start_data_validation(data_ingestion_artifact)
                data_transformation_artifact = self.start_data_transformation(
                    data_ingestion_artifact
                )
                model_trainer_artifact = self.start_model_trainer(
                    data_transformation_artifact
                )

                model_evaluation_artifact = self.start_model_evaluation(
                    data_transformation_artifact,
                    model_trainer_artifact,
                )
                self.start_model_pusher(model_evaluation_artifact)

                logging.info("=" * 50)
                logging.info("Pipeline completed successfully")

            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)
