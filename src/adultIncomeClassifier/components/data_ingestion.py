import os
import gdown
from zipfile import ZipFile
from adultIncomeClassifier.entity import DataIngestionConfig
from adultIncomeClassifier import logger
from adultIncomeClassifier.utils import get_size
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from adultIncomeClassifier.utils import read_yaml
from adultIncomeClassifier.constants import SCHEMA_FILE_PATH


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:

            self.config = config
        except Exception as e:
            raise e

    def download_file(self):
        try:
            logger.info("Trying to download file...")
            if not os.path.exists(self.config.local_data_file):
                logger.info("Download started...")
                filename = gdown.download(    
                    url=self.config.source_URL,
                    output=self.config.local_data_file,
                    quiet=False
                )
                logger.info(f"{filename} downloaded!")
            else:
                logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")        
        except Exception as e:
            raise e

    def unzip_and_save(self):
        try:

            logger.info(f"unzipping file")
            with ZipFile(file=self.config.local_data_file, mode="r") as zf:

                file = zf.namelist()[0]
                target_filepath = os.path.join(Path(self.config.unzip_dir), file)
                   
                if not os.path.exists(target_filepath):
                    zf.extract(file, self.config.unzip_dir)
                        
                    logger.info(f"{file} unziped and saved in {target_filepath}")
                else:
                    logger.info(f"{file} file already exist in {target_filepath}.")
            return target_filepath

        except Exception as e:
            raise e
        
    def split_data_into_train_test(self,target_filepath:Path):
        try:
            dataset_schema = read_yaml(SCHEMA_FILE_PATH)
            target_column = dataset_schema.target_column
            
            df = pd.read_csv(target_filepath)
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
            X = df.drop(target_column,axis=1)
            y = df[target_column]

            for train_index,test_index in sss.split(X,y):
                train_dataset = df.loc[train_index]
                test_dataset = df.loc[test_index]

            train_dataset.to_csv(self.config.train_data_file)
            test_dataset.to_csv(self.config.test_data_file)

            logger.info(f"train data saved in {self.config.train_data_file}.")
            logger.info(f"test data saved in {self.config.test_data_file}.")
        
        except Exception as e:
            raise e