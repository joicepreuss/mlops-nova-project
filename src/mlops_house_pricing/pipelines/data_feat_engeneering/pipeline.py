"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engeneering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_engeneering,
                inputs=["X_train_cleaned", "X_test_cleaned" ],
                outputs=["X_train_transformed", 
                         "X_test_transformed", 
                         "train_transformed_describe",
                         "feat_eng_preprocessor"],
                name="preprocess",
            )
        ]
    )
