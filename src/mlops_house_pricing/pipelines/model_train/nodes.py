
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os

import mlflow
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def model_train(X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_test: pd.DataFrame):
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.

    Returns:
    --
        model (pickle): Trained models.
        scores (json): Trained model metrics.
    """

    # enable autologging
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    logger.info('Starting first step of model selection : Comparing between modes types')
    mlflow.sklearn.autolog()

    # open pickle file with regressors
    with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
        regressor = pickle.load(f)

    results_dict = {}
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        model = regressor.fit(X_train, y_train)
        # making predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        # evaluating model
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        # saving results in dict
        results_dict['regressor'] = regressor.__class__.__name__
        results_dict['train_score'] = mse_train
        results_dict['test_score'] = mse_test
        # logging in mlflow
        run_id = mlflow.last_active_run().info.run_id
        logger.info(f"Logged train model in run {run_id}")
    return model, results_dict