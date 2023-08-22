"""
This is a boilerplate pipeline '03_regression_for_over_threshold'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_modelinput_data, train_model, plot_feature_importance, plot_learning_curve, evaluate_model, predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_modelinput_data,
                inputs=["train_data_over_threshold_for_price_regresstion", "parameters"],
                outputs=["X_train_over_threshold_for_price_regresstion", "y_train_over_threshold_for_price_regresstion"],
                name="create_train_data_over_threshold_for_price_regresstion_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_over_threshold_for_price_regresstion", "y_train_over_threshold_for_price_regresstion", "parameters"],
                outputs=["regressor_over_threshold", "regression_over_threshold_eval_results"],
                name="train_model_over_threshold_for_price_regresstion_node",
            ),
            node(
                func=plot_feature_importance,
                inputs=["regressor_over_threshold", "parameters"],
                outputs=None,
                name="plot_feature_importance_over_threshold_for_price_regresstion_node",
            ),
            node(
                func=plot_learning_curve,
                inputs=["regression_over_threshold_eval_results", "parameters"],
                outputs=None,
                name="plot_learning_curve_over_threshold_for_price_regresstion_node",
            ),
            node(
                func=create_modelinput_data,
                inputs=["valid_data_over_threshold_for_price_regresstion", "parameters"],
                outputs=["X_valid_over_threshold_for_price_regresstion", "y_valid_over_threshold_for_price_regresstion"],
                name="create_valid_data_over_threshold_for_price_regresstion_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor_over_threshold", "X_valid_over_threshold_for_price_regresstion", "y_valid_over_threshold_for_price_regresstion", "boxcox_lambda_over_threshold_for_price_regresstion", "parameters"],
                outputs=None,
                name="evaluate_model_over_threshold_for_price_regresstion_node",
            ),
            node(
                func=predict,
                inputs=["regressor_over_threshold", "test_data", "boxcox_lambda_over_threshold_for_price_regresstion", "parameters"],
                outputs="y_pred_over_threshold",
                name="predict_over_threshold_for_price_regresstion_node",
            ),
        ]
    )
