"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["X_train_data","X_test_data", "parameters"],
                outputs=["X_train_cleaned", 
                         "X_test_cleaned", 
                         "train_cleaned_describe",
                         "cleaning_preprocessor"],
                name="clean",
            )
        ]
    )
