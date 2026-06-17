import os
import sys
import pandas as pd
from src.constants import *
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.utils.main_utils import load_schema


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def validate_column_count(self, df: pd.DataFrame, schema: dict) -> bool:
        expected_count = len(schema["columns"])
        actual_count = df.shape[1]
        if actual_count != expected_count:
            logging.warning(
                f"Column count mismatch: expected {expected_count}, got {actual_count}"
            )
            return False
        logging.info(f"Column count validation passed: {actual_count} columns")
        return True

    def validate_column_names(self, df: pd.DataFrame, schema: dict) -> bool:
        expected_columns = [list(col.keys())[0] for col in schema["columns"]]
        actual_columns = list(df.columns)
        missing = set(expected_columns) - set(actual_columns)
        extra = set(actual_columns) - set(expected_columns)
        if missing:
            logging.warning(f"Missing columns: {missing}")
            return False
        if extra:
            logging.warning(f"Extra columns: {extra}")
            return False
        logging.info("Column names validation passed")
        return True

    def validate_column_dtypes(self, df: pd.DataFrame, schema: dict) -> bool:
        dtype_map = {"str": "object", "float": "float64", "int": "int64"}
        all_valid = True
        for col_def in schema["columns"]:
            col_name = list(col_def.keys())[0]
            expected_dtype = dtype_map.get(list(col_def.values())[0])
            if expected_dtype and col_name in df.columns:
                actual_dtype = str(df[col_name].dtype)
                if actual_dtype != expected_dtype:
                    if actual_dtype == "int64" and expected_dtype == "float64":
                        continue
                    logging.warning(
                        f"Column '{col_name}' dtype mismatch: "
                        f"expected {expected_dtype}, got {actual_dtype}"
                    )
                    all_valid = False
        if all_valid:
            logging.info("Column dtypes validation passed")
        return all_valid

    def validate_domain_values(self, df: pd.DataFrame, schema: dict) -> bool:
        domain_values = schema.get("domain_values", {})
        all_valid = True
        for col, allowed_values in domain_values.items():
            if col in df.columns:
                actual_values = df[col].dropna().unique()
                invalid = [str(v) for v in actual_values if str(v) not in allowed_values and not pd.isna(v)]
                if invalid:
                    logging.warning(
                        f"Column '{col}' has invalid domain values: {invalid}"
                    )
                    all_valid = False
        if all_valid:
            logging.info("Domain values validation passed")
        return all_valid

    def check_missing_values(self, df: pd.DataFrame) -> dict:
        missing_report = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = round((missing_count / len(df)) * 100, 2)
            if missing_count > 0:
                logging.info(f"Column '{col}': {missing_count} missing ({missing_pct}%)")
                missing_report[col] = {"missing_count": int(missing_count), "missing_pct": float(missing_pct)}
        return missing_report

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            schema = load_schema()
            logging.info(f"Loaded schema with {len(schema['columns'])} columns")

            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Loaded train data with shape: {train_df.shape}")

            validation_results = {}

            validation_results["column_count"] = self.validate_column_count(train_df, schema)
            validation_results["column_names"] = self.validate_column_names(train_df, schema)
            validation_results["column_dtypes"] = self.validate_column_dtypes(train_df, schema)
            validation_results["domain_values"] = self.validate_domain_values(train_df, schema)

            missing_report = self.check_missing_values(train_df)
            validation_results["missing_values"] = missing_report

            validation_status = all(
                validation_results[key] for key in ["column_count", "column_names", "column_dtypes", "domain_values"]
            )
            validation_results["validation_status"] = validation_status

            report = {
                "validation_status": validation_status,
                "results": validation_results,
            }

            report_path = self.data_validation_config.report_file_path
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            from src.utils.main_utils import write_yaml
            write_yaml(report_path, report)

            if validation_status:
                logging.info("Data validation completed successfully")
            else:
                logging.warning("Data validation completed with issues")

            return DataValidationArtifact(
                report_file_path=report_path,
                validation_status=validation_status,
            )

        except Exception as e:
            raise MyException(e, sys)
