import os
import json
import pandas as pd

from src.mlops_house_pricing.pipelines.data_quality.nodes import check_data_feature_engineering
from src.mlops_house_pricing.pipelines.data_quality.nodes import check_data_cleaning

def test_check_data_feature_engineering():
    # Create a sample dataframe for testing
    df = pd.DataFrame({
        'num_1': [1, 2, 3, 4, 5],
        'num_2' : [1, 2, 3, 4, 5],
    })

    # Define the parameters
    parameters = {
        'num_quality_ranges': {
            'min': 1,
            'max': 5
        }
    }

    # Call the function to be tested
    check_data_feature_engineering(df, parameters)

    # Check if the validation results file is saved
    validation_results_file_path = '../data/08_reporting/Expectations_reporting/feature_engineered_data_validation_results.json'
    assert os.path.exists(validation_results_file_path)  # The validation results file should exist

    # Check if the errors file is saved
    errors_file_path = '../data/08_reporting/Expectations_reporting/feature_engineered_data_errors.json'
    assert os.path.exists(errors_file_path)  # The errors file should exist

    # Load the validation results and errors from the files
    with open(validation_results_file_path, 'r') as json_file:
        validation_results = json.load(json_file)
    with open(errors_file_path, 'r') as json_file:
        errors = json.load(json_file)

    # Assert that the validation results contain the expected keys
    assert 'results' in validation_results
    assert 'statistics' in validation_results


def test_check_data_cleaning():
    # Create a sample dataframe for testing
    df = pd.DataFrame({
        'numerical__YrSold': [2000, 2001, 2002, 2003, 2004],
        'numerical__YearBuilt': [2000, 2001, 2002, 2003, 2004],
        'categorical__MSZoning': ['A', 'A', 'A', 'A', 'A'],
    })

    # Define the parameters
    parameters = {
        'num_columns': 3,
        'column_list': ['numerical__YrSold', 'numerical__YearBuilt', 'categorical__MSZoning'],
        'categorical_unique_values': {
            'categorical__MSZoning': ['A'],
    }
    }
    # Call the function to be tested
    check_data_cleaning(df, parameters)
    
    folder_path = os.path.join(os.getcwd(), "data", "08_reporting", "Expectations_reporting")

    # Check if the validation results file is saved
    validation_results_file_path = os.path.join(folder_path, "Cleaned_data_validation_results.json")
    assert os.path.exists(validation_results_file_path)  # The validation results file should exist

    # Load the validation results and errors from the files
    with open(validation_results_file_path, 'r') as json_file:
        validation_results = json.load(json_file)
    
    # Assert that the validation results contain the expected keys
    assert 'results' in validation_results
    assert 'statistics' in validation_results
