
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import mlflow

logger = logging.getLogger(__name__)

def model_train(X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_test: pd.DataFrame, 
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

    mlflow.set_tag("mlflow.runName", parameters["run_name"])
    #mlflow.autolog(log_model_signatures=True, log_input_examples=True)

    regressor = [
            # DecisionTreeRegressor(random_state=parameters["random_state"], max_depth=10),
            RandomForestRegressor(n_jobs=parameters["n_jobs"], 
                                  random_state=parameters["random_state"], 
                                  max_depth=parameters["max_depth"],
                                  n_estimators=parameters["n_estimators"],
                                  max_features=parameters["max_features"]),
            # GradientBoostingRegressor(random_state=parameters["random_state"], max_depth=6)
            ]

    models = {}
    scores = pd.DataFrame(columns=["regressor",'train_score', 'test_score'])
    for reg in regressor:
        reg.fit(X_train, y_train)
        # scores[clf.__class__.__name__] = clf.score(X_test, y_test)

        scores.loc[len(scores)] =  [reg.__class__.__name__,
                                    np.round(reg.score(X_train, y_train),2), 
                                    np.round(reg.score(X_test, y_test),2)]
        
        models[reg.__class__.__name__] = reg

    # get best model from models dict according test score in scores df
    best_model = scores.loc[scores['test_score'].idxmax()]['regressor']
    best_model = models[best_model]

    scores_to_dict = scores.to_dict()

    log = logging.getLogger(__name__)
    log.info(scores)
    log.info(f"Best model: {best_model.__class__.__name__}\n" 
             f"Test score: {scores.loc[scores['test_score'].idxmax()]['test_score']}\n" 
             f"Parameters: {best_model.get_params()}")
    # log.info("Model accuracy on test set: %0.2f%%", accuracy * 100)

    return best_model, scores_to_dict