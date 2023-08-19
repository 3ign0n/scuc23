"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_modelinput_data, train_model, plot_feature_importance, evaluate_model, predict, post_process

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_modelinput_data,
                inputs=["train_data", "parameters"],
                outputs=["X_train", "y_train"],
                name="create_train_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "parameters"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=plot_feature_importance,
                inputs=["regressor", "parameters"],
                outputs=None,
                name="plot_feature_importance_node",
            ),
            node(
                func=create_modelinput_data,
                inputs=["valid_data", "parameters"],
                outputs=["X_valid", "y_valid"],
                name="create_valid_data_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_valid", "y_valid", "parameters"],
                outputs=None,
                name="evaluate_model_node",
            ),
            node(
                func=predict,
                inputs=["regressor", "test_data", "parameters"],
                outputs="y_pred",
                name="predict_node",
            ),
            node(
                func=post_process,
                inputs="parameters",
                outputs=None,
                name="post_process",
            )
        ]
    )
