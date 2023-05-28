
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


def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Clean and preprocesses data for modelling. 
    - Cleans data by dropping columns with only NaNs and associated columns.
    - Imputes missing values with mean for numerical features and most frequent for categorical features.
    - Applies a OneHotEncoder to categorical features.
    - Scaling with RobustScaler for numerical features.
    
    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.

    Returns:
    --
        X_train (pd.DataFrame): Trasnformed training features.
        X_test (pd.DataFrame): Transformed test features.
    """
    
    # Drop columns with only NaNs and associated columns.
    cols_with_only_nans = parameters["cols_with_only_nans"]
    associated_cols = parameters["associated_cols"]

    X_train.drop(columns=cols_with_only_nans + associated_cols, inplace=True)
    X_test.drop(columns=cols_with_only_nans + associated_cols, inplace=True)

    # Define categorical and numerical features.
    categorical_features = list(X_train.select_dtypes('object').columns)
    numerical_features = list(X_train.select_dtypes('number').columns)

    # Pipelines for numerical and categorical features.
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('label', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Combine numerical and categorical pipelines.
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])

    X_train = preprocessor.fit_transform(X_train)
    column_names = preprocessor.get_feature_names_out()
    X_train = pd.DataFrame(X_train.toarray(), columns=column_names)
    X_test = pd.DataFrame(preprocessor.transform(X_test).toarray(), 
                      columns=preprocessor.get_feature_names_out())
    
    # Add prefix to column names, to analyse.
    numerical_features = ["numerical__" + col for col in numerical_features]
    
    describe_to_dict_verified = X_train[numerical_features].describe().to_dict()
    
    logger.info(f"The final train dataframe has {len(X_train.columns)} columns.\n"
                f"The final test dataframe has {len(X_test.columns)} columns.")

    return X_train, X_test, describe_to_dict_verified