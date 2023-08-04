from adultIncomeClassifier import logger
from adultIncomeClassifier.entity import DataTransformationConfig
from adultIncomeClassifier.utils import read_yaml,save_object

import os
import pandas  as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
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
    
    def _ReplaceSymbolLabel(self,train_df:pd.DataFrame,test_df:pd.DataFrame,symbol:str,cols:list):
        try:
            # Replacing ? symbol labels in the columns as None values
            for col in cols:
                train_df[col] = train_df[col].replace(symbol,np.nan)
                test_df[col] = test_df[col].replace(symbol,np.nan)

            logger.info(f"Replacement of {symbol} in {cols} with None is done.")
            return train_df,test_df
        except Exception as e:
            raise e
        
    def _EncodingCatFeatures(self,train_df:pd.DataFrame,test_df:pd.DataFrame,):
        try:
            # Encodeing categorical features label into 0 and 1 
            
            for dataframe in [train_df,test_df]:
                for df in [dataframe]:
                    df.loc[df['country'] != ' United-States', 'country'] = 0
                    df.loc[df['country'] == ' United-States', 'country'] =  1
                    df.loc[df['race'] != ' White', 'race'] = 0
                    df.loc[df['race'] == ' White', 'race'] = 1
                    df.loc[df['workclass'] != ' Private', 'workclass'] = 0
                    df.loc[df['workclass'] == ' Private', 'workclass'] = 1
                    df.loc[df['hours-per-week'] <= 40, 'hours-per-week'] = 0
                    df.loc[df['hours-per-week'] > 40, 'hours-per-week'] = 1 
                    df.loc[df['salary'] ==' <=50K','salary'] = 0
                    df.loc[df['salary'] ==' >50K','salary'] = 1

            logger.info("columns:[country,race,workclass,hours-per-week,salary] are encoded into 0 and 1.")
            
            return train_df,test_df
        
        except Exception as e:
            raise e
        
    def _get_data_preprocessing_object(self) -> ColumnTransformer:
        try:

            schema_file_path = self.config.schema_file_path

            dataset_schema = read_yaml(Path(schema_file_path))

            numerical_columns = dataset_schema.numerical_columns
            categorical_columns = dataset_schema.categorical_columns


            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                 ('impute', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore',sparse=False)),
                 ('scaler', StandardScaler(with_mean=False))
            ])

            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")


            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ],
            remainder='passthrough')

            return preprocessing

        except Exception as e:
            raise e

    def initiate_data_transformation(self):
        try:
            # Getting train and test dataset
            train_df,test_df = self._get_train_and_test_df()

            # Replacing special symbol used for none into np.nan
            cols = ['workclass','occupation','country']
            train_df,test_df = self._ReplaceSymbolLabel(train_df=train_df,test_df=test_df,symbol='?',cols=cols)

            # Encoding Categorical Features into 0 and 1 to remove insignificant label in the columns
            train_df,test_df = self._EncodingCatFeatures(train_df=train_df,test_df=test_df)

            # Getting Preprocessing object
            preprocessing_obj = self._get_data_preprocessing_object()

            logger.info('preprocessing object is loaded.')

            transformed_train_data = preprocessing_obj.fit_transform(train_df)
            transformed_test_data = preprocessing_obj.transform(test_df)

            logger.info('train and test successfully transformed by the preprocessing object.')

            # Saving the transformed train data
            with open(self.config.transformed_train_path,'wb') as file_obj:
                np.save(file_obj,transformed_train_data)

            logger.info(f'transformed train data saved at {self.config.transformed_train_path}')
        
            # Saving the transformed test data
            with open(self.config.transformed_test_path,'wb') as file_obj:
                np.save(file_obj,transformed_test_data)

            logger.info(f'transformed test data saved at {self.config.transformed_test_path}')

            # Saving the preprocessing object 
            save_object(preprocessing_obj,file_path=Path(self.config.preprocessed_object_file_path))

            logger.info(f"preprocessing object binary file saved at {self.config.preprocessed_object_file_path}")

        except Exception as e:
            raise e