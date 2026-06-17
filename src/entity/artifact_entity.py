from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path
