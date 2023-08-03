from adultIncomeClassifier import logger
from adultIncomeClassifier.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from adultIncomeClassifier.utils import read_yaml, create_directories
from adultIncomeClassifier.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
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

        try:
            config = self.config.data_validation

            create_directories([config.root_dir])

            data_validation_config = DataValidationConfig(
                root_dir= config.root_dir,
                schema_file_path= config.schema_file_path,
                train_data_file_path= config.train_data_file_path,
                test_data_file_path=config.test_data_file_path,
                report_file_path=config.report_file_path,
                report_page_file_path=config.report_page_file_path
            )

            return data_validation_config
        
        except Exception as e:
            raise e
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            config = self.config.data_transformation

            create_directories([config.root_dir])

            data_transformation_config = DataTransformationConfig(
                root_dir=config.root_dir,
                schema_file_path= config.schema_file_path,
                train_data_file_path=config.train_data_file_path,
                test_data_file_path = config.test_data_file_path,
                transformed_train_path=config.transformed_train_path,
                transformed_test_path=config.transformed_test_path,
                preprocessed_object_file_path=config.preprocessed_object_file_path
            )

            return data_transformation_config
        except Exception as e:
            raise e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            config = self.config.model_trainer

            create_directories([config.root_dir])

            model_trainer_config = ModelTrainerConfig(
                root_dir= config.root_dir,
                transformed_train_path= config.transformed_train_path,
                transformed_test_path= config.transformed_test_path,
                preprocessed_pkl_file_path= config.preprocessed_pkl_file_path,
                model_config_file_path= config.model_config_file_path,
                trained_model_file_path= config.trained_model_file_path,
                base_accuracy= config.base_accuracy
            )

            return model_trainer_config
        except Exception as e:
            raise e
