import os
import sys
from src.exception import MyException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
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

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact
            )
            logging.info("=" * 50)
            logging.info("Pipeline completed successfully")
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys)
