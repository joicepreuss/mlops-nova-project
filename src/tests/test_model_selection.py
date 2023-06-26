import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.mlops_house_pricing.pipelines.model_selection.nodes import model_selection


def test_model_selection():
    """
    Test that the model selection node returns a model with a score
    """
    # Create dummy data
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    X_test = pd.DataFrame(np.random.rand(50, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    y_train = pd.DataFrame(np.random.rand(100), columns=['target'])
    y_test = pd.DataFrame(np.random.rand(50), columns=['target'])
    
    champion_dict = {'regressor': None, 'test_score': 0}
    
    champion_model = None
    
    parameters = {
        'hyperparameters': {
            'RandomForestRegressor': {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]},
            'GradientBoostingRegressor': {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]},
            'LinearRegression': {'alpha': [0, 1]}
        }
    }
    
    # Run the model selection node
    model = model_selection(X_train, X_test, y_train, y_test, champion_dict, champion_model, parameters)
    
    # Check that the returned value is a dictionary
    assert isinstance(model, RandomForestRegressor) or isinstance(model, GradientBoostingRegressor) or isinstance(model, ElasticNet)
    assert isinstance(model.score(X_test, y_test), float)
