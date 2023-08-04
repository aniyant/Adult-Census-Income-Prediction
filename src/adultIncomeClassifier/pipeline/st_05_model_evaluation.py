from adultIncomeClassifier.config import ConfigurationManager
from adultIncomeClassifier.components import ModelEvaluation
from adultIncomeClassifier import logger

STAGE_NAME = "Model Evaluation Stage"

def main():
    config = ConfigurationManager()
    model_evaluation_config = config.get_model_evaluation_config()
    model_evaluation = ModelEvaluation(config=model_evaluation_config)
    model_evaluation.initiate_model_evaluation()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e