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

from src.mlops_house_pricing.pipelines.data_cleaning.nodes import clean_data

dir = os.path.join(os.getcwd(),'src','tests','sample_data')

# reading parameters file
with open('conf/base/parameters.yml') as f:
    parameters = yaml.load(f, Loader=yaml.loader.SafeLoader)

def test_clean_data():
    # Create sample data
    X_train = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, np.nan, np.nan, np.nan],
        'C': ['a', 'b', 'a', 'c'],
        'D': [1.1, 2.2, 3.3, 4.4]
    })

    X_test = pd.DataFrame({
        'A': [5, 6, 7, 8],
        'B': [np.nan, np.nan, np.nan, np.nan],
        'C': ['a', 'b', 'c', 'd'],
        'D': [5.5, 6.6,7.7, 8.8]
    })

    parameters = {
        "cols_with_only_nans": ['B'],
        "associated_cols": []
    }

    # Call the clean_data function
    X_train_clean, X_test_clean, describe_transformers, cleaning_preprocessor = clean_data(X_train, X_test, parameters)

    # Check if the function dropped the correct columns
    assert 'B' not in X_train_clean.columns
    assert 'B' not in X_test_clean.columns

    # Check if the function imputed missing values correctly
    assert not X_train_clean['numerical__A'].isnull().any()
    assert not X_train_clean['categorical__C'].isnull().any()
    assert not X_train_clean['numerical__D'].isnull().any()

    # Check if the function transformed the test data correctly
    assert not X_test_clean['numerical__A'].isnull().any()
    assert not X_test_clean['categorical__C'].isnull().any()
    assert not X_test_clean['numerical__D'].isnull().any()

    # Check if the function returned the correct describe_transformers dictionary
    assert isinstance(describe_transformers, dict)
    assert 'numerical__A' in describe_transformers
    assert 'numerical__D' in describe_transformers

    # Check if the function returned the correct cleaning_preprocessor object
    assert isinstance(cleaning_preprocessor, ColumnTransformer)