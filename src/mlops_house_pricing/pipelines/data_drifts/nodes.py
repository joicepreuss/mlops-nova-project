import logging
from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nannyml as nml
from nannyml.drift.multivariate.data_reconstruction.result import Result
import os

logger = logging.getLogger(__name__)

def check_data_drift(reference : pd.DataFrame, analysis : pd.DataFrame, parameters : Dict[str, Any]):
    """
    Data drift detection.
    - Multivariate data drift
    - Univariate data drift
    - Model performance drift

    Args:
    --
        reference (pd.DataFrame): Reference dataset
        analysis (pd.DataFrame): Analysis dataset
        parameters (Dict[str, Any]): Parameters

    """
    reference = create_timestamp_column(reference, 
                                        column_name_year="YrSold", 
                                        column_name_month="MoSold")
    
    analysis = create_timestamp_column(analysis, 
                                       column_name_year="YrSold", 
                                       column_name_month="MoSold")
    
    feature_columns = parameters["most_important_features"]

    reference = reference[feature_columns + ["timestamp"]]
    analysis = analysis[feature_columns + ["timestamp"]]
    
    multivariat_drift_detected = calculate_drift_multivariat(reference, 
                                                            analysis, 
                                                            feature_column_names=feature_columns,
                                                            timestamp_column_name="timestamp")
    
    if multivariat_drift_detected:
        create_psi_plot(feature_columns, reference, analysis, 'drift')
        raise Exception(
            f"Data Drift detected in the multivariate data drift analysis."
        )
    else:
        create_psi_plot(feature_columns, reference, analysis, 'no_drift')
        logger.info(
            f"No data drift detected in the multivariate data drift analysis."
        )

def create_timestamp_column(df : pd.DataFrame, column_name_year : str, column_name_month : str) -> pd.DataFrame:
    """
    This function creates a new timestamp column using a passed year and month column.
    """
    df['timestamp'] = pd.to_datetime(df[column_name_year].astype(str) + '-' + df[column_name_month].astype(str), format='%Y-%m')
    
    return df


def calculate_drift_multivariat(reference : pd.DataFrame, analysis : pd.DataFrame,
                                 feature_column_names : List[str], timestamp_column_name : str="timestamp") -> None:
    """
    Calculates and plots the multivariant data drift.

    Args:
    --
        reference (pd.DataFrame): Reference dataset
        analysis (pd.DataFrame): Analysis dataset
        feature_column_names (List[str]): List of feature column names
        timestamp_column_name (str): Timestamp column name
    """

    folder_path = os.path.join(os.getcwd(), 'data', '08_reporting', 'data_drifts_reporting', 'multivariative_drift')
    os.makedirs(folder_path, exist_ok=True)

    calc = nml.DataReconstructionDriftCalculator(column_names=feature_column_names,
                                                 timestamp_column_name=timestamp_column_name
                                                 )
    calc.fit(reference)

    results = calc.calculate(analysis)
    analysis_results = results.filter(period='analysis').to_df()
    reference_results = results.filter(period='reference').to_df()

    drift_detected = False
    if analysis_results[('reconstruction_error','alert')].max():
        drift_detected = True
        analysis_results.to_csv(os.path.join(folder_path, 'with_drift_multivariate_analysis_results.csv'))
        reference_results.to_csv(os.path.join(folder_path, 'with_drift_multivariate_reference_results.csv'))

        figure = results.plot()
        file_path = os.path.join(folder_path, 'with_drift_multivariate_drift.html')
        figure.write_html(file_path)
    else:
        analysis_results.to_csv(os.path.join(folder_path, 'without_drift_multivariate_analysis_results.csv'))
        reference_results.to_csv(os.path.join(folder_path, 'without_drift_multivariate_reference_results.csv'))

        figure = results.plot()
        file_path = os.path.join(folder_path, 'without_drift_multivariate_drift.html')
        figure.write_html(file_path)

    return drift_detected

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''
    Code copied from the Practical Lab for data drift from MLOps course.
    Calculate the PSI (population stability index) across all variables.
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

def create_psi_plot(numerical_features, reference, analysis, type):
    """
    Create a plot of the PSI values for each numerical feature
    """
    folder_path = os.path.join(os.getcwd(), 'data', '08_reporting', 'data_drifts_reporting', 'psi_plots')
    os.makedirs(folder_path, exist_ok=True)
    sns.set_style("darkgrid")
    psis_num = []

    #Using the github implementation to compute PSI's numerical features
    for feature_name in numerical_features:
        psi = calculate_psi(reference[feature_name], analysis[feature_name], buckettype='bins', buckets=20, axis=0)
        psis_num.append(psi)
    #Plot each feature's PSI value
    height = psis_num
    bars = numerical_features
    y_pos = np.arange(len(bars))
    plt.barh(y_pos, height)
    plt.axvline(x=0.2,color='red')
    plt.yticks(y_pos, bars)
    plt.xlabel("PSI")
    plt.title("PSI for numerical features")
    plt.ylabel("Features")
    if type == "drift":
        file_path = os.path.join(folder_path, 'with_drift_psi_numerical_features.png')
        plt.savefig(file_path)
        plt.close()
    else:
        file_path = os.path.join(folder_path, 'without_drift_psi_numerical_features.png')
        plt.savefig(file_path)
        plt.close()