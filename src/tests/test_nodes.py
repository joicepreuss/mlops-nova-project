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

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager

from src.mlops_exercise.nodes import split_data, preprocess_data, model_train


dir = os.path.join(os.getcwd(),'src','tests','sample_data')

# reading configuration experiment file
with open('conf/base/parameters.yml') as f:
    parameters = yaml.load(f, Loader=yaml.loader.SafeLoader)

def test_split_data_volumetry():
    df = pd.read_csv(os.path.join(dir,'house-pricing.csv'))
    split_pct = parameters['test_fraction']
    sensitiveness = 0.05
    X_train, X_test, y_train, y_test ,_ = split_data(df,parameters)
    assert (X_test.shape[0] <= df.shape[0]*(split_pct+sensitiveness)) & (X_test.shape[0] >= df.shape[0]*(split_pct-sensitiveness))
    assert (X_train.shape[0] <= df.shape[0]*(1-split_pct+sensitiveness)) & (X_train.shape[0] >= df.shape[0]*(1-split_pct-sensitiveness))
    assert (y_test.shape[0] <= df.shape[0]*(split_pct+sensitiveness)) & (y_test.shape[0] >= df.shape[0]*(split_pct-sensitiveness))
    assert (y_train.shape[0] <= df.shape[0]*(1-split_pct+sensitiveness)) & (y_train.shape[0] >= df.shape[0]*(1-split_pct-sensitiveness))

def test_split_data_X_columns():
    df = pd.read_csv(os.path.join(dir,'house-pricing.csv'))
    X_train, X_test, y_train, y_test, describe_to_dict = split_data(df,parameters)
    assert X_train.shape[1] == X_test.shape[1]

# @pytest.mark.slow
# def test_clean_date_null():
#     df = pd.read_csv("/Users/jaime.kuei/Documents/study-repositories/master/second-semester/mlops/notebooks/kedro/class-bank-example-pytest/src/tests/data/sample_test.csv")
#     df_transformed, describe_to_dict, describe_to_dict_verified = clean_data(df)
#     assert [col for col in df_transformed.columns if df_transformed[col].isnull().any()] == []