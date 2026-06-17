import os
import sys
from src.exception import MyException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
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

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(
                f"Pipeline completed. Train: {data_ingestion_artifact.train_file_path}, "
                f"Test: {data_ingestion_artifact.test_file_path}"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)
