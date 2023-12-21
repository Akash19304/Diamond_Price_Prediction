from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys,os
from dataclasses import dataclass
import pandas as pd 
import numpy as np 

from exception import CustomException
from logger import logging

from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiared')
            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']

            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Initiated')

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OrdinalEncoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info('Error in Data Transformation Object')
            raise CustomException(e,sys)

    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read Train and Test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_column = [target_column_name,'id']

            input_feature_train_df = train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## applying transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Applying preprocessng object on training and testing datasets')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
