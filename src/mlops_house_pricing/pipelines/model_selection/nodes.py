
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import yaml
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import mlflow

logger = logging.getLogger(__name__)

     
def model_selection(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_test: pd.DataFrame,
                    champion_dict: Dict[str, Any],
                    champion_model : pickle.Pickler,
                    parameters: Dict[str, Any]):
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.
        parameters (dict): Parameters defined in parameters.yml.

    Returns:
    --
        models (dict): Dictionary of trained models.
        scores (pd.DataFrame): Dataframe of model scores.
    """
   
    models_dict = {
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'LinearRegression': ElasticNet(alpha = 0)
    }

    initial_results = {}   

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    logger.info('Starting first step of model selection : Comparing between model types')

    for model_name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id,nested=True):
            mlflow.sklearn.autolog()
            model.fit(X_train, y_train)
            initial_results[model_name] = model.score(X_test, y_test)
            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model : {model_name} in run {run_id}")
    
    best_model_name = max(initial_results, key=initial_results.get)
    best_model = models_dict[best_model_name]

    logger.info(f"Best model is {best_model_name} with score {initial_results[best_model_name]}")
    logger.info('Starting second step of model selection : Hyperparameter tuning')

    # Perform hyperparameter tuning with GridSearchCV
    param_grid = parameters['hyperparameters'][best_model_name]
    with mlflow.start_run(experiment_id=experiment_id,nested=True):
        gridsearch = GridSearchCV(best_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        gridsearch.fit(X_train, y_train)
        best_model = gridsearch.best_estimator_

    # # Log all search results in MLFlow
    # for run_index in range(len(cv_results['params'])):
    #     with mlflow.start_run(experiment_id=experiment_id,run_name = str(run_index),nested=True):
    #         mlflow.log_param("folds", gridsearch.cv)

    #         print("Logging parameters")
    #         params = list(gridsearch.param_grid.keys())
    #         for param in params:
    #             mlflow.log_param(param, cv_results["param_%s" % param][run_index])

    #         print("Logging metrics")
    #         for score_name in [score for score in cv_results if "mean_test" in score]:
    #             mlflow.log_metric(score_name, cv_results[score_name][run_index])
    #             mlflow.log_metric(score_name.replace("mean","std"), cv_results[score_name.replace("mean","std")][run_index])

    #         print("Logging model")        
    #         mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name)

    logger.info(f"Hypertunned model score: {-1 * gridsearch.best_score_}")
    pred_score = mean_squared_error(y_test, best_model.predict(X_test))

    if champion_dict['test_score'] < pred_score:
        logger.info(f"New champion model is {best_model_name} with score: {pred_score} vs {champion_dict['test_score']} ")
        return best_model
    else:
        logger.info(f"Champion model is still {champion_dict['model_name']} with score: {champion_dict['test_score']} vs {pred_score} ")
        return champion_model