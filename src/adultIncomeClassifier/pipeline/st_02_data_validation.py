from adultIncomeClassifier.config import ConfigurationManager
from adultIncomeClassifier.components import DataValidation
from adultIncomeClassifier import logger

STAGE_NAME = "Data Validation Stage"

def main():
    config = ConfigurationManager()
    data_validation_config = config.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_dataset_schema()
    data_validation.get_and_save_data_drift_report()
    data_validation.save_data_drift_report_page()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e