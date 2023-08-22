"""
This is a boilerplate pipeline '02_regression_for_under_threshold'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_modelinput_data, train_model, plot_feature_importance, plot_learning_curve, evaluate_model, predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
           node(
                func=create_modelinput_data,
                inputs=["train_data_under_threshold_for_price_regresstion", "parameters"],
                outputs=["X_train_under_threshold_for_price_regresstion", "y_train_under_threshold_for_price_regresstion"],
                name="create_train_data_under_threshold_for_price_regresstion_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_under_threshold_for_price_regresstion", "y_train_under_threshold_for_price_regresstion", "parameters"],
                outputs=["regressor_under_threshold", "regression_under_threshold_eval_results"],
                name="train_model_under_threshold_for_price_regresstion_node",
            ),
            node(
                func=plot_feature_importance,
                inputs=["regressor_under_threshold", "parameters"],
                outputs=None,
                name="plot_feature_importance_under_threshold_for_price_regresstion_node",
            ),
            node(
                func=plot_learning_curve,
                inputs=["regression_under_threshold_eval_results", "parameters"],
                outputs=None,
                name="plot_learning_curve_under_threshold_for_price_regresstion_node",
            ),
            node(
                func=create_modelinput_data,
                inputs=["valid_data_under_threshold_for_price_regresstion", "parameters"],
                outputs=["X_valid_under_threshold_for_price_regresstion", "y_valid_under_threshold_for_price_regresstion"],
                name="create_valid_data_under_threshold_for_price_regresstion_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor_under_threshold", "X_valid_under_threshold_for_price_regresstion", "y_valid_under_threshold_for_price_regresstion", "parameters"],
                outputs=None,
                name="evaluate_model_under_threshold_for_price_regresstion_node",
            ),
            node(
                func=predict,
                inputs=["regressor_under_threshold", "test_data", "parameters"],
                outputs="y_pred_under_threshold",
                name="predict_under_threshold_for_price_regresstion_node",
            ),
        ]
    )
