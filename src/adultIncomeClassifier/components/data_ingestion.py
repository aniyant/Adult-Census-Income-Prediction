import os
import gdown
from zipfile import ZipFile
from adultIncomeClassifier.entity import DataIngestionConfig
from adultIncomeClassifier import logger
from adultIncomeClassifier.utils import get_size
from pathlib import Path


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

        except Exception as e:
            raise e