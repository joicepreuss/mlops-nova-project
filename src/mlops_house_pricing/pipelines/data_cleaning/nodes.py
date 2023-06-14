
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
import pickle

logger = logging.getLogger(__name__)


def clean_data(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pickle.Pickler]:
    """Clean data. 
    - Cleans data by dropping columns with only NaNs and associated columns.
    - Imputes missing values with mean for numerical features and most frequent for categorical features.
    
    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.

    Returns:
    --
        X_train (pd.DataFrame): Trasnformed training features.
        X_test (pd.DataFrame): Transformed test features.
        describe_transformers (Dict): Dictionary with statistics of the transformers.
        preprocessor (pickle.Pickler): Preprocessor object.
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
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])
    # Combine numerical and categorical pipelines.
    cleaning_preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])

    X_train = cleaning_preprocessor.fit_transform(X_train)
    column_names = cleaning_preprocessor.get_feature_names_out()
    # X_train = pd.DataFrame(X_train.toarray(), columns=column_names)
    X_train = pd.DataFrame(X_train, columns=column_names)
    # X_test = pd.DataFrame(preprocessor.transform(X_test).toarray(), 
    #                   columns=preprocessor.get_feature_names_out())
    X_test = pd.DataFrame(cleaning_preprocessor.transform(X_test), 
                      columns=cleaning_preprocessor.get_feature_names_out())
    
    # Add prefix to column names, to analyse.
    numerical_features = ["numerical__" + col for col in numerical_features]
    
    describe_to_dict_verified = X_train[numerical_features].describe().to_dict()
    
    logger.info(f"The final train dataframe has {len(X_train.columns)} columns.\n"
                f"The final test dataframe has {len(X_test.columns)} columns.")

    return X_train, X_test, describe_to_dict_verified, cleaning_preprocessor