import os
import sys

from logger import logging
from exception import CustomException
import pandas as pd
from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model_trainer import ModelTrainer

if __name__=="__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    data_transformation = DataTransformation()

    train_arr, test_arr, _ = data_transformation.initiate_data_tranformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiare_model_training(train_arr, test_arr)