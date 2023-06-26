import logging
from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
import nannyml as nml
from nannyml.drift.multivariate.data_reconstruction.result import Result


logger = logging.getLogger(__name__)



def create_timestamp_column(df : pd.DataFrame, column_name_year : str, column_name_month : str) -> pd.DataFrame:
    """
    This function creates a new timestamp column using a passed year and month column
    """
    df['timestamp'] = pd.to_datetime(df[column_name_year].astype(str) + '-' + df[column_name_month].astype(str), format='%Y-%m')
    
    return df


def filter_rows_by_years(df : pd.DataFrame, years : int, months : int=None,
                         timestamp_column_name : str="timestamp") -> pd.DataFrame:
    """
    This function returns a df with the passed years like in this example
    reference = filter_rows_by_years(df, [2007, 2008])
    analysis = filter_rows_by_years(df, [2009, 2010])
    months can also be defined like:
    first_half_2006 = filter_rows_by_years(df, [2006], [1, 2, 3, 4, 5, 6])
    second_half_2006 = filter_rows_by_years(df, [2006], [7, 8, 9, 10, 11, 12])
    """
    filtered_df = df[df[timestamp_column_name].dt.year.isin(years)]
    
    if months:
        filtered_df = filtered_df[filtered_df[timestamp_column_name].dt.month.isin(months)]
    
    return filtered_df


"""
NOT NEEDED ANYMORE: PERFORMANCE EVALUATION DOESN'T NEED A PROBABILITY!!

def binarize_sales(df : pd.DataFrame) -> pd.DataFrame:
    median_sale_price = df["SalePrice"].median()
    df['binarized'] = df["SalePrice"].apply(lambda x: 1 if x > median_sale_price else 0)

    return df


def create_probability(x_train : pd.DataFrame, x_test : pd.DataFrame, y_true : pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()

    x_train["proba_pred"] = scaler.fit_transform(X=np.array(x_train["SalePrice"]).reshape(-1,1), y=y_true)
    x_test["proba_pred"] = scaler.transform(X=np.array(x_test["SalePrice"]).reshape(-1,1))

    return x_train, x_test, scaler
"""

def calculate_drift_multivariant(reference : pd.DataFrame, analysis : pd.DataFrame,
                                 feature_column_names : List[str], timestamp_column_name : str="timestamp") -> None:
    """
    This function calculates and plots the multivariant data drift
    """

    calc = nml.DataReconstructionDriftCalculator(column_names=feature_column_names,
                                                 timestamp_column_name=timestamp_column_name
                                                 )
    calc.fit(reference)

    results = calc.calculate(analysis)
    display(results.filter(period='analysis').to_df())
    display(results.filter(period='reference').to_df())

    figure = results.plot()
    figure.show()


def calculate_drift_univariante(reference : pd.DataFrame, analysis : pd.DataFrame,
                                column_names : List[str], treat_as_categorical : List[str],
                                timestamp_column_name : str, continuous_methods : List[str]=['kolmogorov_smirnov', 'jensen_shannon'],
                                categorical_methods : List[str]=['chi2', 'jensen_shannon']) -> Result:
    """
    This function calculates and plots the univariante data drift
    """

    calc = nml.UnivariateDriftCalculator(column_names=column_names,
                                         treat_as_categorical=treat_as_categorical,
                                         timestamp_column_name=timestamp_column_name,
                                         continuous_methods=continuous_methods,
                                         categorical_methods=categorical_methods
                                         )
    calc.fit(reference)
    results = calc.calculate(analysis)

    return results

def estimate_performance(reference : pd.DataFrame, analysis : pd.DataFrame,
                         feature_column_names : List[str], y_pred : pd.Series, y_true : pd.Series,
                         timestamp_column_name : str, metrics: List[str]=['rmse', 'rmsle'],
                         tune_hyperparameters : bool=False) -> None:
    
    """
    This function is estimating the model performance using the DLE
    """

    estimator = nml.DLE(feature_column_names=feature_column_names,
                        y_pred=y_pred,
                        y_true=y_true,
                        timestamp_column_name=timestamp_column_name,
                        metrics=metrics,
                        tune_hyperparameters=tune_hyperparameters
                        )
    
    estimator.fit(reference)
    results = estimator.estimate(analysis)
    display(results.filter(period='analysis').to_df())
    display(results.filter(period='reference').to_df())
    metric_fig = results.plot()
    metric_fig.show()


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


def calculate_psi_categorical(actual, expected):
    actual_perc = actual.value_counts()/len(actual)
    expected_perc = expected.value_counts()/len(expected)
    actual_classes = list(actual_perc.index) 
    expected_classes = list(expected_perc.index)
    PSI = 0
    classes = set(actual_classes + expected_classes)
    for c in classes:
        final_actual_perc = actual_perc[c] if c in actual_classes else 0.00001
        final_expected_perc = expected_perc[c] if c in expected_classes else 0.00001
        PSI += (final_actual_perc - final_expected_perc)*np.log(final_actual_perc/final_expected_perc)
    return PSI