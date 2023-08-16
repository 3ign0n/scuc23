"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import set_random_state, enable_autologging, preprocess_train_data, preprocess_test_data, save_pandas_profiling, preprocess_do_label_encoding


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=set_random_state,
                inputs="parameters",
                outputs=None,
                name="set_random_state_node",
            ),
            node(
                func=enable_autologging,
                inputs="parameters",
                outputs=None,
                name="enable_autologging_node",
            ),
            node(
                func=preprocess_train_data,
                inputs=["train_data", "region_state_data"],
                outputs="preprocessed_train_data",
                name="preprocess_train_data_node",
            ),
            node(
                func=preprocess_test_data,
                inputs=["test_data", "region_state_data"],
                outputs="preprocessed_test_data",
                name="preprocess_test_data_node",
            ),
            node(
                func=save_pandas_profiling,
                inputs=["preprocessed_train_data", "preprocessed_test_data"],
                outputs=None,
                name="save_pandas_profiling_node",
            ),
            node(
                func=preprocess_do_label_encoding,
                inputs="preprocessed_train_data",
                outputs="encoded_train_data",
                name="preprocess_train_data_label_encoding_node",
            ),
            node(
                func=preprocess_do_label_encoding,
                inputs="preprocessed_test_data",
                outputs="encoded_test_data",
                name="preprocess_test_data_label_encoding_node",
            ),
    ])
