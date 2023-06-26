"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import check_data_drift

def create_pipeline(type, **kwargs) -> Pipeline:
    if type == 'data_drift':
        return pipeline(
            [
                node(
                    func=check_data_drift,
                    inputs=["reference_raw", "house_pricing_raw", "parameters"],
                    outputs=None,
                    name="data_drift",
                )
            ]
        )
    elif type == 'simulate_data_drift':
        return pipeline(
            [
                node(
                    func=check_data_drift,
                    inputs=["reference_raw", "drift_data_raw", "parameters"],
                    outputs=None,
                    name="simulate_data_drift",
                )
            ]
        )