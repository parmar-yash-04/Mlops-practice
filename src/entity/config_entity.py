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
