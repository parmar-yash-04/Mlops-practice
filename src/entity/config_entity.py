from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir: Path
    feature_store_path: Path
    ingested_dir: Path
    train_path: Path
    test_path: Path
    test_size: float


@dataclass
class DataValidationConfig:
    root_dir: Path
    report_file_path: Path


@dataclass
class DataTransformationConfig:
    root_dir: Path
    transformed_train_dir: Path
    transformed_test_dir: Path
    preprocessing_obj_path: Path


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    trained_model_dir: Path
    trained_model_path: Path
    expected_score: float
    model_config_file_path: Path


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    trained_model_path: Path
    changed_threshold_score: float


@dataclass
class ModelPusherConfig:
    root_dir: Path
    trained_model_path: Path
    preprocessing_obj_path: Path
