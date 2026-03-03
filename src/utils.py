import os
import sys
from lightgbm import cv
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
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

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        logging.info("Evaluating models.")
        model_report = {}
        for name, model in models.items():
            if name in params:
                gs= GridSearchCV(model, params[name], cv=3, n_jobs=-1, verbose=2,refit=True)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[name] = r2
        logging.info("Model evaluation completed successfully.")
        return model_report
    except Exception as e:
        logging.error("Error occurred during model evaluation.")
        raise CustomException(e, sys)