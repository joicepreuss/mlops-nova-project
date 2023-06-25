import mlflow
import pandas as pd

from src.mlops_house_pricing.pipelines.model_train.nodes import model_train

def test_model_train():
    # Create dummy data for testing
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    X_test = pd.DataFrame({'feature1': [7, 8, 9], 'feature2': [10, 11, 12]})
    y_train = pd.DataFrame({'target': [2, 4, 6]})
    y_test = pd.DataFrame({'target': [8, 10, 12]})
    parameters = {
        "run_name": "test_run",
        "n_jobs": -1,
        "max_depth": 6,
        "random_state": 42,
        "n_estimators": 100,
        "max_features": 20
    }
    mlflow.set_experiment(parameters["run_name"])
    # Call the function to train the model
    best_model, scores_to_dict = model_train(X_train, X_test, y_train, y_test, parameters)

    # Assertions
    assert isinstance(scores_to_dict, dict)
    assert "regressor" in scores_to_dict
    assert "train_score" in scores_to_dict
    assert "test_score" in scores_to_dict
    assert best_model is not None
    assert best_model.__class__.__name__ in scores_to_dict["regressor"][0]
