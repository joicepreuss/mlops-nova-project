import logging
from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import nannyml as nml
from nannyml.drift.multivariate.data_reconstruction.result import Result
import os

logger = logging.getLogger(__name__)

def check_data_drift(reference : pd.DataFrame, analysis : pd.DataFrame, parameters : Dict[str, Any]):

# 1. timestamp column, wird kreiert aus dem month sold and year sold
# 2. reference = dataset mit prediction und cleaned und y_true
# 3. analysis = mock dataset ohne prediction und cleaned
# 4. 2 features distribution verändern
# reference dataframe = golden dataset mit test und predictions und y_true (aus raw und predictions)

    reference = create_timestamp_column(reference, 
                                        column_name_year="YrSold", 
                                        column_name_month="MoSold")
    
    analysis = create_timestamp_column(analysis, 
                                       column_name_year="YrSold", 
                                       column_name_month="MoSold")
    
    feature_columns = parameters["most_important_features"]

    reference = reference[feature_columns + ["timestamp", "y_pred", "y_true"]]
    analysis = analysis[feature_columns + ["timestamp"]]
    
    calculate_psi(reference, analysis, buckettype='bins', buckets=10, axis=0)

    calculate_drift_multivariat(reference, 
                                 analysis, 
                                 feature_column_names=feature_columns,
                                 timestamp_column_name="timestamp")
    
    calculate_drift_univariate(reference, 
                                analysis, 
                                column_names=feature_columns, 
                                treat_as_categorical=[], 
                                timestamp_column_name="timestamp")
    
    estimate_performance(reference,
                        analysis,
                        feature_column_names=feature_columns,
                        y_pred=reference["y_pred"],
                        y_true=reference["y_true"],
                        timestamp_column_name="timestamp",
                        metrics=['rmse', 'rmsle'],
                        tune_hyperparameters=False)
    
    create_psi_plot(feature_columns, reference, analysis)



def create_timestamp_column(df : pd.DataFrame, column_name_year : str, column_name_month : str) -> pd.DataFrame:
    """
    This function creates a new timestamp column using a passed year and month column
    """
    df['timestamp'] = pd.to_datetime(df[column_name_year].astype(str) + '-' + df[column_name_month].astype(str), format='%Y-%m')
    
    return df


def calculate_drift_multivariat(reference : pd.DataFrame, analysis : pd.DataFrame,
                                 feature_column_names : List[str], timestamp_column_name : str="timestamp") -> None:
    """
    This function calculates and plots the multivariant data drift
    """

    folder_path = '../data/08_reporting/Data_drifts_reporting/Multivariate_drifts'
    os.makedirs(folder_path, exist_ok=True)

    calc = nml.DataReconstructionDriftCalculator(column_names=feature_column_names,
                                                 timestamp_column_name=timestamp_column_name
                                                 )
    calc.fit(reference)

    results = calc.calculate(analysis)
    analysis_results = results.filter(period='analysis').to_df()
    reference_results = results.filter(period='reference').to_df()

    analysis_results.to_csv(os.path.join(folder_path, 'Multivariate_analysis_results.csv'))
    reference_results.to_csv(os.path.join(folder_path, 'Multivariate_reference_results.csv'))

    figure = results.plot()
    file_path = os.path.join(folder_path, 'multivariate_drift.html')
    figure.write_html(file_path)


def calculate_drift_univariate(reference : pd.DataFrame, analysis : pd.DataFrame,
                                column_names : List[str], treat_as_categorical : List[str],
                                timestamp_column_name : str, continuous_methods : List[str]=['kolmogorov_smirnov', 'jensen_shannon'],
                                categorical_methods : List[str]=['chi2', 'jensen_shannon']) -> Result:
    """
    This function calculates and plots the univariate data drift
    """

    folder_path = '../data/08_reporting/Data_drifts_reporting/Univariate_drifts'
    os.makedirs(folder_path, exist_ok=True)

    calc = nml.UnivariateDriftCalculator(column_names=column_names,
                                         treat_as_categorical=treat_as_categorical,
                                         timestamp_column_name=timestamp_column_name,
                                         continuous_methods=continuous_methods,
                                         categorical_methods=categorical_methods
                                         )
    calc.fit(reference)
    results = calc.calculate(analysis)

    analysis_results = results.filter(period='analysis').to_df()
    reference_results = results.filter(period='reference').to_df()

    analysis_results.to_csv(os.path.join(folder_path, 'Univariate_analysis_results.csv'))
    reference_results.to_csv(os.path.join(folder_path, 'Univariate_reference_results.csv'))

    jensen = results.filter(column_names=results.continuous_column_names, methods=['jensen_shannon']).plot(kind='drift')
    file_path_jensen = os.path.join(folder_path, 'Univariate_drift_jensen_shannon.html')
    jensen.write_html(file_path_jensen)
    
    kolgomorov = results.filter(column_names=results.continuous_column_names, methods=['kolmogorov_smirnov']).plot(kind='drift')
    file_path_kolgomorov = os.path.join(folder_path, 'Univariate_drift_kolgomorov_smirnov.html')
    kolgomorov.write_html(file_path_kolgomorov)

    return results

def estimate_performance(reference : pd.DataFrame, analysis : pd.DataFrame,
                         feature_column_names : List[str], y_pred : pd.Series, y_true : pd.Series,
                         timestamp_column_name : str, metrics: List[str]=['rmse', 'rmsle'],
                         tune_hyperparameters : bool=False) -> None: 
    """
    This function is estimating the model performance using the DLE
    """
    folder_path = '../data/08_reporting/Data_drifts_reporting/Estimate_performance'
    os.makedirs(folder_path, exist_ok=True)

    estimator = nml.DLE(feature_column_names=feature_column_names,
                        y_pred=y_pred,
                        y_true=y_true,
                        timestamp_column_name=timestamp_column_name,
                        metrics=metrics,
                        tune_hyperparameters=tune_hyperparameters
                        )
    
    estimator.fit(reference)
    results = estimator.estimate(analysis)

    analysis_results = results.filter(period='analysis').to_df()
    reference_results = results.filter(period='reference').to_df()
    analysis_results.to_csv(os.path.join(folder_path, 'Estimate_performance_analysis_results.csv'))
    reference_results.to_csv(os.path.join(folder_path, 'Estimate_performance_reference_results.csv'))

    metric_fig = results.plot()
    file_path = os.path.join(folder_path, 'estimate_performance.html')
    metric_fig.write_html(file_path)

# CODE FOR PSI FROM LAB1

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
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

def create_psi_plot(numerical_features, reference, analysis):
    folder_path = '../data/08_reporting/Data_drifts_reporting'
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
    file_path = os.path.join(folder_path, 'psi_numerical_features.png')
    plt.savefig(file_path)


# def calculate_psi_categorical(actual, expected):
#     actual_perc = actual.value_counts()/len(actual)
#     expected_perc = expected.value_counts()/len(expected)
#     actual_classes = list(actual_perc.index) 
#     expected_classes = list(expected_perc.index)
#     PSI = 0
#     classes = set(actual_classes + expected_classes)
#     for c in classes:
#         final_actual_perc = actual_perc[c] if c in actual_classes else 0.00001
#         final_expected_perc = expected_perc[c] if c in expected_classes else 0.00001
#         PSI += (final_actual_perc - final_expected_perc)*np.log(final_actual_perc/final_expected_perc)
#     return PSI