import pandas as pd
import numpy as np

from src.mlops_house_pricing.pipelines.data_split.nodes import split_data

def test_split_data():
    """Test the split_data function.
    """

    # Create a sample DataFrame
    data = {
        'Id': [1, 2, 3, 4, 5],
        'Feature1': [10, 20, 30, 40, 50],
        'Feature2': [100, 200, 300, 400, 500],
        'Target': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Define the parameters
    parameters = {
        'target_column': 'Target',
        'random_state': 42,
        'test_fraction': 0.2
    }

    # Call the split_data function
    X_train, X_test, y_train, y_test, describe_dict = split_data(df, parameters)

    # Define the expected outputs
    expected_X_train = pd.DataFrame({'Feature1': [50, 30, 10, 40], 
                                     'Feature2': [500, 300, 100, 400]}, index=[4, 2, 0, 3])
    expected_X_test = pd.DataFrame({'Feature1': [20], 'Feature2': [200]}, index=[1])
    expected_y_train = pd.Series([1, 1, 1, 0], name='Target', index=[4, 2, 0, 3])
    expected_y_test = pd.Series([0], name='Target', index=[1])

    expected_describe = df.describe().to_dict()

    # Assert the existence of the datasets
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert describe_dict is not None

    # Assert the shapes of the resulting datasets
    assert X_train.shape == (4, 2)
    assert X_test.shape == (1, 2)
    assert y_train.shape == (4,)
    assert y_test.shape == (1,)

    # Assert the content of the resulting datasets
    assert np.allclose(X_train, expected_X_train)
    assert np.allclose(X_test, expected_X_test)
    assert np.allclose(y_train, expected_y_train)
    assert np.allclose(y_test, expected_y_test)

    # Assert the content of the describe_dict
    assert describe_dict == expected_describe

