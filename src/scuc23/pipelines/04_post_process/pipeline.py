"""
This is a boilerplate pipeline '04_post_process'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import ensemble


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=ensemble,
                inputs=["y_pred_under_threshold", "y_pred_over_threshold", "highend_proba", "parameters"],
                outputs="y_pred",
                name="ensemble_node",
            ),
        ]
    )
