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
