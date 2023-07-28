from adultIncomeClassifier import logger
from adultIncomeClassifier.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from adultIncomeClassifier.utils import read_yaml, create_directories
from adultIncomeClassifier.entity import (
    DataIngestionConfig,
    DataValidationConfig,
)

from pathlib import Path
import os



class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config.data_ingestion
            
            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir,
                train_data_file=config.train_data_file,
                test_data_file=config.test_data_file
            )

            return data_ingestion_config
        except Exception as e:
            raise e
        
    def get_data_validation_config(self) -> DataValidationConfig:
        pass
