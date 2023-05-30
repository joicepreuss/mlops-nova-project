
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
        gdf.expect_column_values_to_be_between(column,ranges['min'],ranges['max'])
    
    validation_results = gdf.validate()
    failed_expectations = [result for result in validation_results["results"] if not result["success"]]
    
    logger.info(
        f"Total Expectations: {len(validation_results['results'])}"
        f"Failed Expectations: {len(failed_expectations)}"
    )
    
    if failed_expectations:
        collect_errors = []
        for idx, failed_expectation in enumerate(failed_expectations, start=1):
            collect_errors.append(
                f"\nFailed Expectation {idx}:"
                f"  Expectation Type: {failed_expectation['expectation_config']['expectation_type']}"
                f"  Column: {failed_expectation['expectation_config']['kwargs']['column']}"
                f"  Details: {failed_expectation['result']}")
    
        raise Exception(
            f"Data Quality Validation Failed: {collect_errors}"
        )
   
    return df


