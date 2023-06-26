
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import pickle


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

logger = logging.getLogger(__name__)

def model_predict(X: pd.DataFrame,
                  cleaning_preprocessor: pickle.Pickler,
                  feat_eng_preprocessor: pickle.Pickler,
                  model: pickle.Pickler) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict using the trained model.

    Args:
    --
        X (pd.DataFrame): Serving observations.
        cleaning_preprocessor (pickle): Preprocessor for cleaning.
        feat_eng_preprocessor (pickle): Preprocessor for feature engeneering.
        model (pickle): Trained model.

    Returns:
    --
        scores (pd.DataFrame): Dataframe with new predictions.
    """
    # Cleaning
    X =  pd.DataFrame(cleaning_preprocessor.transform(X), 
                      columns=cleaning_preprocessor.get_feature_names_out())
    # Feature engeneering
    X = pd.DataFrame(feat_eng_preprocessor.transform(X), 
                      columns=feat_eng_preprocessor.get_feature_names_out())

    # Predict
    y_pred = model.predict(X)

    # Create dataframe with predictions
    X['y_pred'] = y_pred
    
    # Create dictionary with predictions
    describe_servings = X.describe().to_dict()

    logger.info('Service predictions created.')
    logger.info('#servings: %s', len(y_pred))
    return X, describe_servings