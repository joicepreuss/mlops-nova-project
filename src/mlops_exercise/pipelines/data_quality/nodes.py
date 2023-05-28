
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import great_expectations as ge

import mlflow

logger = logging.getLogger(__name__)



def check_ranges(df: pd.DataFrame, parameters : Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
    """Check for set of itens in categorcial variables.
    Args:
    --
        df (pd.DataFrame): Dataframe to check for nulls.
    Returns:
    --
        df (pd.DataFrame): Dataframe with nulls removed.
        describe_to_dict (dict): Description of the dataframe.
    """
    
    num_cols = df.select_dtypes(include=['number']).columns
    ranges = parameters["num_quality_ranges"]
    gdf = ge.from_pandas(df)
    for column in num_cols:
        print(column)
        gdf.expect_column_values_to_be_between(column,ranges['min'],ranges['max'])
    return df, gdf.validate()


