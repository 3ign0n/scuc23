"""
This is a boilerplate pipeline '00_data_preprocessing'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import set_random_state, enable_autologging, preprocess_data, save_pandas_profiling, preprocess_do_dummy_encoding, preprocess_split_train_data, preprocess_split_train_data_for_price_classification, preprocess_split_train_data_by_price_threshold, preprocess_split_train_data_with_boxcox


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
                inputs=["preprocessed_train_data", "preprocessed_test_data", "parameters"],
                outputs=None,
                name="save_pandas_profiling_node",
            ),
            node(
                func=preprocess_do_dummy_encoding,
                inputs=["preprocessed_train_data", "parameters"],
                outputs="encoded_train_data",
                name="preprocess_train_dummy_encoding_node",
            ),
            node(
                func=preprocess_do_dummy_encoding,
                inputs=["preprocessed_test_data", "parameters"],
                outputs="test_data",
                name="preprocess_test_dummy_encoding_node",
            ),
            node(
                func=preprocess_split_train_data_for_price_classification,
                inputs=["encoded_train_data", "parameters"],
                outputs=["train_data_for_price_classification", "valid_data_for_price_classification"],
                name="preprocess_split_train_data_for_price_classification_node",
            ),
            node(
                func=preprocess_split_train_data_by_price_threshold,
                inputs=["encoded_train_data"],
                outputs=["train_data_under_threshold", "train_data_over_threshold"],
                name="preprocess_split_train_data_by_price_threshold_node",
            ),
            node(
                func=preprocess_split_train_data,
                inputs=["train_data_under_threshold", "parameters"],
                outputs=["train_data_under_threshold_for_price_regresstion", "valid_data_under_threshold_for_price_regresstion"],
                name="preprocess_split_train_data_under_threshold_for_regresstion_node",
            ),
            node(
                func=preprocess_split_train_data_with_boxcox,
                inputs=["train_data_over_threshold", "parameters"],
                outputs=["train_data_over_threshold_for_price_regresstion", "valid_data_over_threshold_for_price_regresstion", "boxcox_lambda_over_threshold_for_price_regresstion"],
                name="preprocess_split_train_data_over_threshold_for_regresstion_node",
            ),
    ])
