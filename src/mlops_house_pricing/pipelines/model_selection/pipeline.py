"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=["X_train_transformed",
                        "X_test_transformed",
                        "y_train_data",
                        "y_test_data",
                        "production_model",
                        "production_model_metrics",
                        "parameters"],
                outputs=["champion_model", "champion_model_metrics"],
                name="model_selection",
            ),
        ]
    )
