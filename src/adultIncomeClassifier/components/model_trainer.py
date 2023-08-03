from adultIncomeClassifier import logger
from adultIncomeClassifier.entity import ModelTrainerConfig
from adultIncomeClassifier.entity import (
    ModelFactory,
    GridSearchedBestModel,
    MetricInfoArtifact,
    evaluate_classification_model )
from adultIncomeClassifier.utils import load_object,save_object
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List


class IncomeClassifierModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

class ModelTrainer:

    def __init__(self, config:ModelTrainerConfig):
        try:
            self.config = config
        except Exception as e:
            raise e
    
    def _get_train_test_data(self,):
        try:
            train_file_path = self.config.transformed_train_path
            test_file_path = self.config.transformed_test_path
            
            if os.path.exists(path=train_file_path) and os.path.exists(path=test_file_path):
                train_data = np.load(train_file_path,allow_pickle=True)
                test_data = np.load(test_file_path,allow_pickle=True)

                logger.info(f"train_data:{train_file_path} and test_data:{test_file_path} are loaded successfully.")
            else:
                logger.info(f"train_data:{train_file_path} and test_data:{test_file_path} not found.")
           
            train_data = train_data.astype('int32')
            test_data = test_data.astype('int32')
            
            return train_data, test_data
        except Exception as e:
            raise e
    
    def initiate_model_training(self):
        try:
            train_data, test_data = self._get_train_test_data()

            # Spliting of train data and test data
            x_train, y_train, x_test, y_test = train_data[:,:-1],train_data[:,-1],test_data[:,:-1],test_data[:,-1]

            logger.info(f"Splitting of training and testing input and target feature are done.")

            logger.info(f"Extracting model config file path")
            model_config_file_path = self.config.model_config_file_path

            logger.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)

            base_accuracy = self.config.base_accuracy
            logger.info(f"Expected accuracy: {base_accuracy}")

            logger.info(f"Initiating operation model selecttion")
            best_model = model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=base_accuracy)
            
            logger.info(f"Best model found on training dataset: {best_model}")
            
            logger.info(f"Extracting trained model list.")
            grid_searched_best_model_list:List[GridSearchedBestModel]=model_factory.grid_searched_best_model_list
            
            model_list = [model.best_model for model in grid_searched_best_model_list ]
            logger.info(f"Evaluation all trained model on training and testing dataset both")
            metric_info:MetricInfoArtifact = evaluate_classification_model(model_list=model_list,X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,base_accuracy=base_accuracy)

            logger.info(f"Best found model on both training and testing dataset.{metric_info.model_object}")
            
            preprocessing_obj=  load_object(file_path=Path(self.config.preprocessed_pkl_file_path))
            model_object = metric_info.model_object

            
            trained_model_file_path=self.config.trained_model_file_path
            income_classifier_model = IncomeClassifierModel(preprocessing_object=preprocessing_obj,trained_model_object=model_object)
            logger.info(f"Saving model at path: {trained_model_file_path}")
            save_object(income_classifier_model,file_path=Path(trained_model_file_path))



            logger.info(f"Model metric info: {metric_info}")
        
        except Exception as e:
            raise e
    