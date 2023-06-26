"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import check_data_feature_engineering, check_data_cleaning

def create_pipeline(type, **kwargs) -> Pipeline:
    if type == 'feature_engineering':
        return pipeline(
            [
                node(
                    func=check_data_feature_engineering,
                    inputs=["X_train_transformed", "parameters"],
                    outputs=None,
                    name="data_quality_feature_engineering",
                )
            ]
        )
    elif type == 'data_clean':
        return pipeline(
            [
                node(
                    func=check_data_cleaning,
                    inputs=["X_train_cleaned", "parameters"],
                    outputs=None,
                    name="data_quality_cleaned_data",
                )
            ]
        )