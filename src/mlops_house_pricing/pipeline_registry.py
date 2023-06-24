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
from mlops_house_pricing.data_quality.nodes import check_ranges


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    split_pipe = ds.create_pipeline()
    cleaning_pipe = dc.create_pipeline()
    feature_eng_pipe = df.create_pipeline()
    model_train_pipe = dm.create_pipeline()
    model_predict_pipe = dpred.create_pipeline()

    check_ranges_after_scaling_pipe = Pipeline(
        [
            node(
                check_ranges,
                inputs=["X_train_transformed","parameters"],
                outputs=[]
            )
        ]
    )

    return {
        "predict": model_predict_pipe,
        "train": split_pipe + cleaning_pipe + feature_eng_pipe + check_ranges_after_scaling_pipe + model_train_pipe,
        #"model_selection" : split_pipe + cleaning_pipe + feature_eng_pipe + model_selection_pipe + hyperparameter_tuning_pipe,
        #"process": split_pipe + cleaning_pipe, Is it necessary?
        "_default_": split_pipe + cleaning_pipe + feature_eng_pipe + model_train_pipe + model_predict_pipe,
    }