
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import great_expectations as ge
import datetime
import json
import os

logger = logging.getLogger(__name__)    

# def check_ranges(df: pd.DataFrame, parameters : Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
#     """Check for set of itens in categorcial variables.
#     Args:
#     --
#         df (pd.DataFrame): Dataframe to check for nulls.
#     Returns:
#     --
#         df (pd.DataFrame): Dataframe with nulls removed.
#         describe_to_dict (dict): Description of the dataframe.
#     """
    
#     num_cols = df.select_dtypes(include=['number']).columns
#     ranges = parameters["num_quality_ranges"]
#     gdf = ge.from_pandas(df)
#     for column in num_cols:
#         gdf.expect_column_values_to_be_between(column,ranges['min'],ranges['max'])
#         gdf.expect_table_column_count_to_equal(81)
#         gdf.expect_column_values_to_not_be_null(column)
    
#     validation_results = gdf.validate()
#     failed_expectations = [result for result in validation_results["results"] if not result["success"]]
    
#     logger.info(
#         f"Ranges considered: {ranges}\n"
#         f"Total Expectations: {len(validation_results['results'])}\n"
#         f"Failed Expectations: {len(failed_expectations)}\n"
#     )
    
#     if failed_expectations:
#         collect_errors = []
#         for idx, failed_expectation in enumerate(failed_expectations, start=1):
#             collect_errors.append(
#                 f"  Failed Expectation {idx}:"
#                 f"  Expectation Type: {failed_expectation['expectation_config']['expectation_type']}"
#                 f"  Column: {failed_expectation['expectation_config']['kwargs']['column']}"
#                 f"  %Instances with errors: {failed_expectation['result']['unexpected_percent_total']}"
#                 f"  Instances with errors: {failed_expectation['result']['partial_unexpected_list']}")
    
#         raise Exception(
#             f"Data Quality Validation Failed: {collect_errors}"
#         )
   
#     return df


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

def check_expectations(df: pd.DataFrame, parameters : Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
    """
    Check expectations for the dataset.
    - Check if the number of columns.
    - Check if the columns exist.
    - Check if the columns are of the correct type.
    - Check if the categorical columns have the correct unique values.
    - Check if the numeric columns are within the correct range.
    - Check if the columns have null values.
    - Check if the ID has only unique values.
    - Check if the median of the SalePrice is within the threshold.
    - Check if the YearBuilt and YrSold are within the correct range.

    Afterwards save the validation results and raise an exception and save the errors, if any of the expectations fail.
    Args:
    --
        df (pd.DataFrame): Dataframe to check for nulls.
    Returns:
    --
        df (pd.DataFrame): Input dataframe.
    """
    current_year = datetime.date.today().year + 1
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_columns = parameters["num_columns"]
    column_list = parameters["column_list"]
    cat_unique_values = parameters["categorical_unique_values"]
    median_sales_price = parameters["median_sales_price"]
    median_threshold = parameters["median_threshold"]

    ranges = parameters["num_quality_ranges"]
    gdf = ge.from_pandas(df)

    gdf.expect_table_column_count_to_equal(num_columns)
    check_if_column_exist(gdf, column_list)
    check_dtype(gdf, num_cols, dtype='numeric')
    check_dtype(gdf, cat_cols, dtype='object')
    check_categorical_unique_values(gdf, cat_unique_values)
    
    gdf.expect_column_median_to_be_between("SalePrice", 
                                           median_sales_price*(1-median_threshold), 
                                           median_sales_price*(1+median_threshold))
    
    check_nulls(gdf, gdf.columns)
    gdf.expect_column_values_to_be_unique("Id")

    for column in num_cols:
        gdf.expect_column_values_to_be_between(column,ranges['min'],ranges['max'])

    gdf.expect_column_max_to_be_between("YearBuilt", 1800, current_year)
    gdf.expect_column_max_to_be_between("YrSold", 1950, current_year)
    
    validation_results = gdf.validate()
    validation_results.to_json_dict("../data/08_reporting/Expectations_validation_results.json")

    failed_expectations = [result for result in validation_results["results"] if not result["success"]]
    
    logger.info(
        f"Total Expectations: {len(validation_results['results'])}"
        f"Failed Expectations: {len(failed_expectations)}"
    )
    
    if failed_expectations:
        collect_errors = []
        for idx, failed_expectation in enumerate(failed_expectations, start=1):
            collect_errors.append(
                f"  Failed Expectation {idx}:"
                f"  Expectation Type: {failed_expectation['expectation_config']['expectation_type']}"
                f"  Column: {failed_expectation['expectation_config']['kwargs']['column']}"
                f"  Details: {failed_expectation['result']}")
            
            # Saves the collected errors in a json file.
            folder_path = '../data/08_reporting'
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, 'Expectations_errors.json')
            with open(file_path, 'w') as json_file:
                json.dump(collect_errors, json_file)
    
        raise Exception(
            f"Data Quality Validation Failed: {collect_errors}"
        )
      
    return df



