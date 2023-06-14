
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


def feature_engeneering(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pickle.Pickler]:
    """Feature engeneering data for modelling. 
    - Applies a OneHotEncoder to categorical features.
    - Scaling with RobustScaler for numerical features.
    
    Args:
    --
        X_train_cleaned (pd.DataFrame): Training features.
        X_test_cleaned (pd.DataFrame): Test features.

    Returns:
    --
        X_train_transformed (pd.DataFrame): Trasnformed training features.
        X_test_transformed (pd.DataFrame): Transformed test features.
        describe (Dict): Dictionary with statistics of transformed dataset.
        preprocessor (pickle.Pickler): Preprocessor object.
    """

    # Define categorical and numerical features.
    categorical_features = list(X_train.select_dtypes('object').columns)
    numerical_features = list(X_train.select_dtypes('number').columns)

    # Pipelines for numerical and categorical features.
    numerical_pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])
    categorical_pipeline = Pipeline([
        ('label', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Combine numerical and categorical pipelines.
    feat_eng_preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])

    X_train = feat_eng_preprocessor.fit_transform(X_train)
    column_names = feat_eng_preprocessor.get_feature_names_out()
    X_train = pd.DataFrame(X_train.toarray(), columns=column_names)
    X_test = pd.DataFrame(feat_eng_preprocessor.transform(X_test).toarray(), 
                      columns=feat_eng_preprocessor.get_feature_names_out())
    
    # Add prefix to column names, to analyse.
    numerical_features = ["numerical__" + col for col in numerical_features]
    
    describe_to_dict_verified = X_train[numerical_features].describe().to_dict()
    
    logger.info(f"The final train dataframe has {len(X_train.columns)} columns.\n"
                f"The final test dataframe has {len(X_test.columns)} columns.")

    return X_train, X_test, describe_to_dict_verified, feat_eng_preprocessor