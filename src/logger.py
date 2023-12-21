import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" ## filename
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE) ## path of logs folder
os.makedirs(logs_path, exist_ok=True) ## create logs folder if it doesn't exist

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE) ## path of log file

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)