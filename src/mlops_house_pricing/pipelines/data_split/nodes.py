
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import mlflow

logger = logging.getLogger(__name__)


def split_data(data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
    --
        data (pd.DataFrame): Data containing features and target.
        parameters (dict): Parameters defined in parameters.yml.
    Returns:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
    """

    df_transformed = data.copy()
    describe_to_dict = df_transformed.describe().to_dict()

    target_column = parameters["target_column"]
    random_state = parameters["random_state"]
    test_fraction = parameters["test_fraction"]

    target = df_transformed[target_column]
    df_transformed = df_transformed.drop(columns=["Id", target_column])

    X_train, X_test, y_train, y_test = train_test_split(df_transformed, 
                                                        target, 
                                                        test_size=test_fraction,
                                                        random_state=random_state)
    # assert [col for col in data.columns if data[col].isnull().any()] == []

    logger.info(f"X_train shape: {X_train.shape}'\n" +
                f"X_test shape: {X_test.shape}\n" +
                f"Test fraction: {test_fraction}")

    return X_train, X_test, y_train, y_test, describe_to_dict