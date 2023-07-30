"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_train_data, preprocess_test_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=preprocess_train_data,
                inputs="train_data",
                outputs="preprocessed_train_data",
                name="preprocess_train_data_node",
            ),
            node(
                func=preprocess_test_data,
                inputs="test_data",
                outputs="preprocessed_test_data",
                name="preprocess_test_data_node",
            ),
    ])
