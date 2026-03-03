import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import save_object,evaluate_models
from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.data_transformation import DataTransformationConfig, DataTransformation

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.model_trainer_config = config

    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        logging.info("Model Trainer method starts")
        try:
            # Simulating model training process
            logging.info("Model training is in progress...")
            # Here you can add code to train a machine learning model using the provided training and testing data.
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Ada Boost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(),
                "CatBoost Classifier": CatBoostRegressor()
            }

            params={
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "Ada Boost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },
                "XGBoost": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "KNN": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },
                "SVR": {
                    'kernel': ['linear', 'rbf'],
                    'C': [1, 10]
                },
                "CatBoost Classifier": {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'depth': [3, 5]
                }

            }

            model_report :dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,params=params)
            

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 Score greater than 0.6", sys)
            
            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with R2 Score: {best_model_score}")
            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")


            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model)
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"Model training completed with R2 Score: {r2_square}")
            logging.info(f"Model training is completed successfully with best model : {best_model_name} and R2 Score: {best_model_score}")
            return r2_square
        
        except Exception as e:
            logging.error("Error occurred during Model Training.")
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    config = ModelTrainerConfig()
    obj = ModelTrainer(config)
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')
    data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(data_transformation_config)
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    obj.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
    print("Model training completed with R2 Score: ", obj.initiate_model_trainer(train_arr, test_arr, preprocessor_path))