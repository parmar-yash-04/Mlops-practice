from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path


@dataclass
class DataValidationArtifact:
    report_file_path: Path
    validation_status: bool


@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    preprocessing_obj_file_path: Path
