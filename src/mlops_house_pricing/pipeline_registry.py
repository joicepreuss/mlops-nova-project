#"""Project pipelines."""
#from typing import Dict

#from kedro.framework.project import find_pipelines
#from kedro.pipeline import Pipeline


#def register_pipelines() -> Dict[str, Pipeline]:
#    """Register the project's pipelines.

#    Returns:
#        A mapping from pipeline names to ``Pipeline`` objects.
#    """
#    pipelines = find_pipelines()
#    pipelines["_default_"] = sum(pipelines.values())
#    return pipelines


"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, node

from mlops_house_pricing.pipelines import data_split as ds
from mlops_house_pricing.pipelines import data_cleaning as dc
from mlops_house_pricing.pipelines import data_feat_engeneering as df
from mlops_house_pricing.pipelines import model_train as dm
from mlops_house_pricing.pipelines import model_predict as dpred
from mlops_house_pricing.pipelines import model_selection as dms
from mlops_house_pricing.pipelines import data_quality as dq
from mlops_house_pricing.pipelines import data_drifts as data_drifts


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    split_pipe = ds.create_pipeline()
    cleaning_pipe = dc.create_pipeline()
    feature_eng_pipe = df.create_pipeline()

    data_quality_feature_engineering = dq.create_pipeline('feature_engineering')
    data_quality_cleaned_data = dq.create_pipeline('data_clean')
    
    model_train_pipe = dm.create_pipeline()
    model_predict_pipe = dpred.create_pipeline()
    model_selection_pipe = dms.create_pipeline()

    data_drift = data_drifts.create_pipeline('data_drift')
    data_drift_simulation = data_drifts.create_pipeline('simulate_data_drift')

    return {
        "predict": model_predict_pipe,
        "train": data_drift + split_pipe + cleaning_pipe + data_quality_cleaned_data + feature_eng_pipe + data_quality_feature_engineering + model_train_pipe,
        "simulate_drift": data_drift_simulation + split_pipe + cleaning_pipe + data_quality_cleaned_data + feature_eng_pipe + data_quality_feature_engineering + model_train_pipe,
        "model_selection" : data_drift + split_pipe + cleaning_pipe + data_quality_cleaned_data + feature_eng_pipe + data_quality_feature_engineering + model_selection_pipe,
        "__default__": data_drift + split_pipe + cleaning_pipe + data_quality_cleaned_data + feature_eng_pipe + data_quality_feature_engineering + model_train_pipe + model_predict_pipe,
    }