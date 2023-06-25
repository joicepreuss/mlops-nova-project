"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path
import sys
import os
import yaml

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager

from src.mlops_house_pricing.pipelines.data_feat_engeneering.nodes import feature_engeneering

dir = os.path.join(os.getcwd(),'src','tests','sample_data')

def test_feature_engeneering():
    # Create sample data
    data_train = {'numerical_feature': [1, 2, 3, 4], 'categorical_feature': ['A', 'B', 'A', 'A']}
    data_test = {'numerical_feature': [4, 5], 'categorical_feature': ['B','A']}
    
    X_train = pd.DataFrame(data_train)
    X_test = pd.DataFrame(data_test)
    
    # Call the feature_engeneering function
    X_train_transformed, X_test_transformed, describe, preprocessor = feature_engeneering(X_train, X_test)
    
    # Check the output types
    assert isinstance(X_train_transformed, pd.DataFrame)
    assert isinstance(X_test_transformed, pd.DataFrame)
    assert isinstance(describe, dict)
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Check the transformed data shape
    assert X_train_transformed.shape[0] == X_train.shape[0]
    assert X_test_transformed.shape[0] == X_test.shape[0]
    
    # Check the transformed data
    expected_columns = ['numerical__numerical_feature', 'categorical__categorical_feature_A', 'categorical__categorical_feature_B']
    assert list(X_train_transformed.columns) == expected_columns
    assert list(X_test_transformed.columns) == expected_columns