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
    root_dir: Path
    schema_file_path: Path
    train_data_file_path: Path
    test_data_file_path: Path
    report_file_path: Path
    report_page_file_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    schema_file_path: Path
    train_data_file_path: Path
    test_data_file_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    preprocessed_object_file_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    transformed_train_path: Path
    transformed_test_path: Path
    preprocessed_pkl_file_path: Path
    model_config_file_path: Path
    trained_model_file_path: Path
    base_accuracy: float
    model_config_file_path: Path
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    trained_model_file_path: Path
