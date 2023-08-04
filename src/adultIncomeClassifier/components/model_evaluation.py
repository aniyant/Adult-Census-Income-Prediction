from adultIncomeClassifier import logger
from adultIncomeClassifier.entity import ModelEvaluationConfig
from adultIncomeClassifier.components import IncomeClassifierModel
from adultIncomeClassifier.utils import load_object,save_object

import numpy as np
import os
from pathlib import Path
import pandas as pd

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig) -> None:
        try:
            self.config = config
        except Exception as e:
            raise e
        
    def _get_trained_model(self,):
        try:
            trained_model_path = self.config.trained_model_file_path

            if os.path.exists(path=trained_model_path):
                trained_model = load_object(file_path=Path(trained_model_path))
                logger.info(f"trained model from loaded succuessfully.")
            else:
                logger.info("trained model not found.")
            
            return trained_model
        except Exception as e:
            raise e
        
    def saving_evaluated_model(self,):
        try:
            pass
        except Exception as e:
            raise e 

    def initiate_model_evaluation(self,):
        try:
            trained_model = self._get_trained_model()

            test_data = [[39,' Male', 77516, ' Bachelors', ' Never-married', ' Adm-clerical',' Not-in-family', '1', '0', 2174, 0, 0, '1', 12, '0']]
            logger.info(f"Testing model working on {test_data},")
            test_df = pd.DataFrame(test_data, columns = ['age', 'sex', 'fnlwgt', 'education', 
                                           'marital-status', 'occupation', 'relationship', 
                                           'race', 'workclass', 'capital-gain', 
                                           'capital-loss', 'hours-per-week', 'country',
                                           'education-num','salary'
                                    ])
            prediction = trained_model.predict(test_df)
            if prediction[0] == 1:
                prediction_output = "Income is >50k."
            else:
                prediction_output = "Income is <50k."
            logger.info(f"model can predict so working fine here is prediction:{prediction}:{prediction_output}")
        except Exception as e:
            raise e
        
