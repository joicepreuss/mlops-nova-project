"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  clean_data#, feature_engineer, split_data, model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="abst_work_data",
                outputs="abst_work_cleaned_data",#,"raw_describe","cleaned_describe"],
                name="clean",
            ),
        ]
    )
