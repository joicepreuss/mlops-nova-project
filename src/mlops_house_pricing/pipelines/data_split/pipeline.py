"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["house_pricing_raw", "parameters"],
                outputs=["X_train_data",
                         "X_test_data",
                         "y_train_data",
                         "y_test_data",
                         "train_raw_describe"],
                name="split",
            )
        ]
    )
