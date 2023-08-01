"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_train_test_data, train_model, evaluate_model, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_train_test_data,
                inputs=["preprocessed_train_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="create_train_test_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "X_test", "y_test", "parameters"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test", "parameters"],
                outputs=None,
                name="evaluate_model_node",
            ),
            node(
                func=predict,
                inputs=["regressor", "preprocessed_test_data", "parameters"],
                outputs="y_pred",
                name="predict_node",
            ),
        ]
    )