import sys, os
# from exception import CustomException
from logger import logging
# from utils import load_object
import pandas as pd
import pickle


def error_message_detail(error, error_detail:sys ):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured during load object')
        raise CustomException(e,sys)
    


class PredictPipeline:
    def __init__(self):
        pass

    def prediction(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info('Exception occured during prediction')
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,carat:float,depth:float,table:float,
                 x:float,y:float,z:float,
                 cut:str, color:str, clarity:str):
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Created')
            return df
        except Exception as e:
            logging.info('Exception duing dataframe creation')
            raise CustomException(e,sys)