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
from kedro.pipeline import Pipeline, pipeline

from mlops_house_pricing.pipelines import data_modeling as dm
from mlops_house_pricing.pipelines import data_processing as dp
from mlops_house_pricing.data_quality.nodes import check_ranges


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_modeling_pipeline = dm.create_pipeline()


    return {
        "preprocessing": preprocessing_stage,
        "split_data": split_data_stage,
        "train": train_stage,
        "feature_selection": feature_selection_stage,
        "predict": predict_stage,
        "drift_test" : drift_test_stage, 
        "_default_": preprocessing_stage + split_data_stage + train_stage
    }