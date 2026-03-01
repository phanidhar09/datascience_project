import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config


    def initiate_data_ingestion(self,config: DataIngestionConfig):
        logging.info("Data Ingestion method starts")
        try:
            # Simulating data ingestion process
            logging.info("Data Ingestion is in progress...")
            # Here you can add code to read data from a source, e.g., CSV, database, etc.
            df= pd.read_csv("notebook/data/data.csv")
            logging.info("Data read successfully from source.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.train_data_path, index=False)
            df.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data saved successfully to the specified paths.")

            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion is completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred during Data Ingestion.")
            raise CustomException(e, sys)
    
if __name__ == "__main__":
    config = DataIngestionConfig()
    obj = DataIngestion(config)
    obj.initiate_data_ingestion(config)