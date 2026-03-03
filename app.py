from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import numpy as np


from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_train import ModelTrainerConfig, ModelTrainer

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.pipeline.predict_pipeline import predict_pipeline,customData

app = Flask(__name__)
@app.route('/train', methods=['POST'])
def train_model():
    return render_template('index.html')

@app.route('/predict_data', methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = customData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        try:
            pred_df = data.get_data_as_dataframe()
            predict_pipeline_obj = predict_pipeline()
            result = predict_pipeline_obj.predict(features=pred_df)
            return render_template('home.html', result=result[0])
        except Exception as e:
            logging.error("Error occurred during prediction.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)