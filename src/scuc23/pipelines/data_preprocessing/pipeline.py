"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import set_random_state, enable_autologging, preprocess_data, save_pandas_profiling, preprocess_do_label_encoding, split_train_data


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
                func=preprocess_data,
                inputs=["raw_train_data"],
                outputs="preprocessed_train_data",
                name="preprocess_train_data_node",
            ),
            node(
                func=preprocess_data,
                inputs=["raw_test_data"],
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
                outputs="test_data",
                name="preprocess_test_data_label_encoding_node",
            ),
            node(
                func=split_train_data,
                inputs=["encoded_train_data", "parameters"],
                outputs=["train_data", "valid_data"],
                name="split_train_data_node",
            ),
    ])
