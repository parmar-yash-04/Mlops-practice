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
