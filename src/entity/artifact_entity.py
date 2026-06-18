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


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: Path
    train_accuracy: float
    test_accuracy: float
    model_accuracy: float


@dataclass
class ModelEvaluationArtifact:
    model_accepted: bool
    changed_accuracy: float
    trained_model_path: Path
    test_accuracy: float


@dataclass
class ModelPusherArtifact:
    model_pushed: bool
    model_registry_path: str
