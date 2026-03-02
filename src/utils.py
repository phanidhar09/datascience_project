import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.exception import CustomException
from src.logger import logging
def save_object(file_path, obj):
    try:
        logging.info("Saving object to file: %s", file_path)
        # Here you can add code to save the object to a file, e.g., using pickle or joblib.
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info("Object saved successfully.")
    except Exception as e:
        logging.error("Error occurred while saving object to file.")
        raise CustomException(e, sys)
