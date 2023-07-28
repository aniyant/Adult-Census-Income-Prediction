from adultIncomeClassifier.config import ConfigurationManager
from adultIncomeClassifier.components import DataIngestion
from adultIncomeClassifier import logger

STAGE_NAME = "Data Ingestion stage"

def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    target_file_path = data_ingestion.unzip_and_save()
    data_ingestion.split_data_into_train_test(target_filepath=target_file_path)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e