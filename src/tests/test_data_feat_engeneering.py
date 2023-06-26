import pandas as pd

from sklearn.compose import ColumnTransformer
from src.mlops_house_pricing.pipelines.data_feat_engeneering.nodes import feature_engeneering

def test_feature_engeneering():
    """
    Test the feature_engeneering function.
    """
    
    # Generate train data
    train_data = {
        'num_1': [1, 2, 3, 4, 5],
        'num_2' : [6, 7, 8, 9, 10],
        'cat_1': ['cat', 'dog', 'cat', 'dog', 'mouse'],
        'cat_2': ['red', 'blue', 'red', 'blue', 'green']
    }
    X_train = pd.DataFrame(train_data)
    
    # Generate test data
    test_data = {
        'num_1': [6, 7, 8],
        'num_2': [11, 12, 13],
        'cat_1': ['dog', 'cat', 'dog'],
        'cat_2': ['blue', 'green', 'red']
    }
    X_test = pd.DataFrame(test_data)
    
    # Expected columns
    expected_columns = ['numerical__num_1', 'numerical__num_2',
                        'categorical__cat_1_cat', 'categorical__cat_1_dog', 
                        'categorical__cat_1_mouse', 'categorical__cat_2_blue', 
                        'categorical__cat_2_green', 'categorical__cat_2_red']
    
    # Call feature_engeneering function
    X_train_transformed, X_test_transformed, describe, preprocessor = feature_engeneering(X_train, X_test)
    
    # Check the output types
    assert isinstance(X_train_transformed, pd.DataFrame)
    assert isinstance(X_test_transformed, pd.DataFrame)
    assert isinstance(describe, dict)
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Check the transformed data shape
    assert X_train_transformed.shape[0] == X_train.shape[0]
    assert X_test_transformed.shape[0] == X_test.shape[0]
    
    # Check the transformed data columns
    assert list(X_train_transformed.columns) == expected_columns
    assert list(X_test_transformed.columns) == expected_columns