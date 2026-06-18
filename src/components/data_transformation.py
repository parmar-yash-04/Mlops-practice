import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from src.constants import *
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.utils.main_utils import save_object, save_numpy_array, load_schema


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def get_preprocessing_pipeline(self) -> Pipeline:
        try:
            schema = load_schema()
            numerical_columns = schema["numerical_columns"]
            categorical_columns = schema["categorical_columns"]

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("label_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns),
                ]
            )
            logging.info("Preprocessing pipeline created")
            return preprocessor
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("=" * 50)
            logging.info("Starting Data Transformation")

            schema = load_schema()
            target_column = schema["target_column"]
            drop_columns = schema.get("drop_columns", [])
            categorical_columns = schema["categorical_columns"]

            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Loaded train data: {train_df.shape}")
            logging.info(f"Loaded test data: {test_df.shape}")

            logging.info("Creating engineered features")
            for df in [train_df, test_df]:
                df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
                df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] / 12)
                df["BalanceIncome"] = df["TotalIncome"] - (df["EMI"] * 1000)
                df["Loan_Income_Ratio"] = df["LoanAmount"] / (df["TotalIncome"] + 1)
            logging.info("Feature engineering completed")

            input_feature_train_df = train_df.drop(columns=[target_column] + drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]
            logging.info(f"Train input features shape: {input_feature_train_df.shape}")

            input_feature_test_df = test_df.drop(columns=[target_column] + drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]
            logging.info(f"Test input features shape: {input_feature_test_df.shape}")

            logging.info("Mapping target column: Y -> 1, N -> 0")
            target_feature_train_df = target_feature_train_df.map({"Y": 1, "N": 0})
            target_feature_test_df = target_feature_test_df.map({"Y": 1, "N": 0})

            logging.info("Creating preprocessing pipeline")
            preprocessing_obj = self.get_preprocessing_pipeline()

            logging.info("Fitting preprocessing pipeline on train data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info(f"Transformed train input shape: {input_feature_train_arr.shape}")

            logging.info("Transforming test data")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info(f"Transformed test input shape: {input_feature_test_arr.shape}")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            os.makedirs(transformed_train_dir, exist_ok=True)
            os.makedirs(transformed_test_dir, exist_ok=True)

            train_file_path = os.path.join(transformed_train_dir, TRAIN_FILE_NAME.replace(".csv", ".npy"))
            test_file_path = os.path.join(transformed_test_dir, TEST_FILE_NAME.replace(".csv", ".npy"))

            save_numpy_array(train_file_path, train_arr)
            save_numpy_array(test_file_path, test_arr)
            logging.info(f"Transformed train saved: {train_file_path}")
            logging.info(f"Transformed test saved: {test_file_path}")

            preprocessing_obj_path = self.data_transformation_config.preprocessing_obj_path
            save_object(preprocessing_obj_path, preprocessing_obj)
            logging.info(f"Preprocessing object saved: {preprocessing_obj_path}")

            logging.info("Data Transformation completed successfully")

            return DataTransformationArtifact(
                transformed_train_file_path=train_file_path,
                transformed_test_file_path=test_file_path,
                preprocessing_obj_file_path=preprocessing_obj_path,
            )

        except Exception as e:
            raise MyException(e, sys)
