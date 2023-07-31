from adultIncomeClassifier import logger
from adultIncomeClassifier.entity import DataValidationConfig
from adultIncomeClassifier.utils import read_yaml,create_directories

import os,sys
import pandas  as pd
from pathlib import Path
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json

class DataValidation:
    def __init__(self, config:DataValidationConfig):
        try:
            self.config = config
        except Exception as e:
            raise e

    def _get_train_and_test_df(self):
        try:
            if os.path.exists(self.config.train_data_file_path):
                train_df = pd.read_csv(self.config.train_data_file_path)

            if os.path.exists(self.config.test_data_file_path):
                test_df = pd.read_csv(self.config.test_data_file_path)

            return train_df,test_df
        except Exception as e:
            raise e
        
    def validate_dataset_schema(self)->bool:
        try:
            validation_status = False

            train_df,test_df = self._get_train_and_test_df()

            schema_file_path = self.config.schema_file_path
            dataset_schema = read_yaml(Path(schema_file_path))

            #validate training and testing dataset using schema file
            #1. Number of Column
            valid_number_of_columns = False
        
            if len(train_df.columns) == len(test_df.columns):
                if len(train_df.columns) == dataset_schema.number_of_cols:
                    valid_number_of_columns = True
            logger.info(f"train:{len(train_df.columns)}, test:{len(test_df.columns)}, schema:{dataset_schema.number_of_cols}")

            logger.info(f"validation of number of columns in train and test  data is : {valid_number_of_columns}")    

            #2. Check column names
            valid_columns_names = False

            for col1 in train_df.columns:
                for col2 in test_df.columns:
                    if col1 == col2:
                        if col1 in dataset_schema.columns.keys():
                            valid_columns_names = True

            logger.info(f"validation names of columns in train and test data is :{valid_columns_names}")

            if valid_columns_names and valid_number_of_columns:
                return validation_status 
            logger.info(f'Validation status of the dataset : {validation_status}')
        except Exception as e:
            raise e
        

    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])

            train_df,test_df = self._get_train_and_test_df()

            profile.calculate(train_df,test_df)

            report = json.loads(profile.json())

            report_file_path = self.config.report_file_path
        
            with open(report_file_path,"w") as report_file:
                json.dump(report, report_file, indent=6)
    
            logger.info(f"Data drift report in json format is saved at : {report_file_path}")
        
        except Exception as e:
            raise e

    def save_data_drift_report_page(self):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df,test_df = self._get_train_and_test_df()
            dashboard.calculate(train_df,test_df)

            report_page_file_path = self.config.report_page_file_path
            
            dashboard.save(report_page_file_path)

            logger.info(f"Data drift report of html page format is saved at : {report_page_file_path}")
        except Exception as e:
            raise e
