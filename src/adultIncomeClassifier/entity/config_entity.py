from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    train_data_file: Path
    test_data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    schema_file_path: Path
    report_file_path: Path
