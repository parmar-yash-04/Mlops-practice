import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.constants import *
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.proj1_data import Proj1Data


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_data_to_feature_store(self) -> pd.DataFrame:
        try:
            proj1_data = Proj1Data()
            df = proj1_data.export_table_as_dataframe(
                table_name=DATA_INGESTION_TABLE_NAME
            )
            proj1_data.close()
            feature_store_path = self.data_ingestion_config.feature_store_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
            df.to_csv(feature_store_path, index=False)
            logging.info(f"Data exported to feature store: {feature_store_path}")
            return df
        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame) -> DataIngestionArtifact:
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=self.data_ingestion_config.test_size,
                random_state=42
            )
            train_path = self.data_ingestion_config.train_path
            test_path = self.data_ingestion_config.test_path
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            logging.info(f"Train data saved: {train_path}")
            logging.info(f"Test data saved: {test_path}")
            return DataIngestionArtifact(
                train_file_path=train_path,
                test_file_path=test_path
            )
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Starting data ingestion")
        df = self.export_data_to_feature_store()
        logging.info(f"Exported data with shape: {df.shape}")
        artifact = self.split_data_as_train_test(df)
        logging.info("Data ingestion completed")
        return artifact
