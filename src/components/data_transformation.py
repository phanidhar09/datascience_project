import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

from src.components.data_ingestion import DataIngestionConfig, DataIngestion


@dataclass
class DataTransformationConfig:
    transformed_train_data_path: str = os.path.join('artifacts', 'transformed_train.csv')
    transformed_test_data_path: str = os.path.join('artifacts', 'transformed_test.csv')
    preprocessor_object_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.data_transformation_config = config

    def get_data_transformer_object(self):
        try:
            # Simulating the creation of a data transformer object
            logging.info("Creating data transformer object.")
            # Here you can add code to create and return a data transformer object, e.g., a ColumnTransformer or a Pipeline.
            numerical_features = ["writing_score","reading_score"]  # Placeholder for actual numerical feature names
            categorical_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]  # Placeholder for actual categorical feature names
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                ('scaler', StandardScaler(with_mean=False))
            ])
            preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ])
            logging.info("Data transformer object created successfully.")
            return preprocessor  # Return the actual preprocessor object
        except Exception as e:
            logging.error("Error occurred while creating data transformer object.")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation method starts")
        try:
            # Simulating data transformation process
            logging.info("Data Transformation is in progress...")
            # Here you can add code to read the train and test data, perform transformations, and save the transformed data.
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data read successfully.")

            preprocessor_obj= self.get_data_transformer_object()

            train_df.columns = train_df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_").str.lower()
            test_df.columns = test_df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_").str.lower()

            target_column = "math_score"  # Placeholder for the actual target column name
            numerical_colums = ["writing_score","reading_score"]  # Placeholder for actual numerical feature names
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]  # Placeholder for actual categorical feature names

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying transformations to the training and testing data.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Transformations applied successfully to the training and testing data.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_object_path,
                obj=preprocessor_obj
            )
            logging.info("Preprocessor object saved successfully.")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_object_path)
        
        except Exception as e:
            logging.error("Error occurred during Data Transformation.")
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    config = DataTransformationConfig()
    obj = DataTransformation(config)
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')
    obj.initiate_data_transformation(train_path, test_path)
