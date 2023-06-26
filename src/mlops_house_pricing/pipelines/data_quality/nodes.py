
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import great_expectations as ge
import datetime
import json
import os

logger = logging.getLogger(__name__)    

# CODE FOR DATA FEATURE ENGINEERING EXPECTATIONS
def check_data_feature_engineering(df: pd.DataFrame, parameters : Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
    """
    Check expectations for the feature engineered dataset.
    - Check if the numerical features are in the expected range.
    - Check if the onehotencoded categorical features have values 0 or 1.

    Afterwards save the validation results and raise an exception and save the errors, if any of the expectations fail.

    Args:
    --
        df (pd.DataFrame): Dataframe to check for nulls.
        parameters (Dict): Parameters from the configuration file.

    Returns:
    --
        df (pd.DataFrame): Input dataframe.
    """
    
    df = df.copy()

    folder_path = '../data/08_reporting/Expectations_reporting'
    os.makedirs(folder_path, exist_ok=True)
    
    cat_cols = df.select_dtypes(include=['object']).columns

    ohencoded_values = [0,1]
    
    gdf = ge.from_pandas(df)
    for column in cat_cols:
        gdf.expect_column_values_to_be_in_set(column, ohencoded_values)

    # Create the validation results and save them in a json file.
    validation_results = gdf.validate()
    file_path_validation_results = os.path.join(folder_path, "feature_engineered_data_validation_results.json")
    with open(file_path_validation_results, 'w') as json_file:
        json.dump(validation_results.to_json_dict(), json_file)

    failed_expectations = [result for result in validation_results["results"] if not result["success"]]
    
    logger.info(
        f"Total Expectations: {len(validation_results['results'])}"
        f"Failed Expectations: {len(failed_expectations)}"
    )
    
    # Collects the errors in a list and saves them in a json file.
    # Afterwards raises an exception with the errors.
    if failed_expectations:
        collect_errors = []
        for idx, failed_expectation in enumerate(failed_expectations, start=1):
            collect_errors.append(
                f"  Failed Expectation {idx}:"
                f"  Expectation Type: {failed_expectation['expectation_config']['expectation_type']}"
                f"  Column: {failed_expectation['expectation_config']['kwargs']['column']}"
                f"  Details: {failed_expectation['result']}")
            
            # Saves the collected errors in a json file.
            file_path = os.path.join(folder_path, 'feature_engineered_data_errors.json')
            with open(file_path, 'w') as json_file:
                json.dump(collect_errors, json_file)
    
        raise Exception(
            f"Data Quality Validation Failed: {collect_errors}"
        )
   

# CODE FOR DATA CLEANING EXPECTATIONS
def check_nulls(gdf, columns):
    for column in columns:
        gdf.expect_column_values_to_not_be_null(column)

def check_categorical_unique_values(gdf, dict_cat_cols):
    for column in dict_cat_cols.keys():
        gdf.expect_column_values_to_be_in_set(column, dict_cat_cols[column])

def check_dtype(gdf, columns, dtype):
    if dtype == 'numeric':
        for column in columns:
            gdf.expect_column_values_to_be_in_type_list(column, ['int64', 'float64'])
    else:
        for column in columns:
            gdf.expect_column_values_to_be_in_type_list(column, ["str"])

def check_if_column_exist(gdf, column_list):
    for column in column_list:
        gdf.expect_column_to_exist(column)

def check_data_cleaning(df: pd.DataFrame, parameters : Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
    """
    Check expectations for the cleaned dataset.
    - Check if the number of columns.
    - Check if the columns exist.
    - Check if the columns are of the correct type.
    - Check if the categorical columns have the correct unique values.
    - Check if the numeric columns are within the correct range.
    - Check if the columns have null values.
    - Check if the YearBuilt and YrSold are within the correct range.

    Afterwards save the validation results and raise an exception and save the errors, if any of the expectations fail.

    Args:
    --
        df (pd.DataFrame): Dataframe to check for nulls.
        parameters (Dict): Parameters from the configuration file.

    Returns:
    --
        df (pd.DataFrame): Input dataframe.
    """

    # Creates a folder to save the expectations results.
    folder_path = '../data/08_reporting/Expectations_reporting'
    os.makedirs(folder_path, exist_ok=True)

    current_year = datetime.date.today().year + 1
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_columns = parameters["num_columns"]
    column_list = parameters["column_list"]
    cat_unique_values = parameters["categorical_unique_values"]

    gdf = ge.from_pandas(df)

    gdf.expect_table_column_count_to_equal(num_columns)
    check_if_column_exist(gdf, column_list)
    check_dtype(gdf, num_cols, dtype='numeric')
    check_dtype(gdf, cat_cols, dtype='object')
    check_categorical_unique_values(gdf, cat_unique_values)
    
    check_nulls(gdf, gdf.columns)
    gdf.expect_column_max_to_be_between("numerical__YearBuilt", 1800, current_year)
    gdf.expect_column_max_to_be_between("numerical__YrSold", 1950, current_year)
    
    # Create the validation results and save them in a json file.
    validation_results = gdf.validate()
    file_path_validation_results = os.path.join(folder_path, "Cleaned_data_validation_results.json")
    with open(file_path_validation_results, 'w') as json_file:
        json.dump(validation_results.to_json_dict(), json_file)

    failed_expectations = [result for result in validation_results["results"] if not result["success"]]
    
    logger.info(
        f"Total Expectations: {len(validation_results['results'])}"
        f"Failed Expectations: {len(failed_expectations)}"
    )
    
    # Collects the errors in a list and saves them in a json file.
    # Afterwards raises an exception with the errors.
    if failed_expectations:
        collect_errors = []
        for idx, failed_expectation in enumerate(failed_expectations, start=1):
            collect_errors.append(
                f"  Failed Expectation {idx}:"
                f"  Expectation Type: {failed_expectation['expectation_config']['expectation_type']}"
                f"  Column: {failed_expectation['expectation_config']['kwargs']['column']}"
                f"  Details: {failed_expectation['result']}")
            
            # Saves the collected errors in a json file.
            file_path = os.path.join(folder_path, 'Cleaned_data_errors.json')
            with open(file_path, 'w') as json_file:
                json.dump(collect_errors, json_file)
    
        raise Exception(
            f"Data Quality Validation Failed: {collect_errors}"
        )
      


