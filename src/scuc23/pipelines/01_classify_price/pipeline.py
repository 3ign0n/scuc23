"""
This is a boilerplate pipeline '01_classify_price'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_modelinput_data, train_model, plot_feature_importance, plot_learning_curve, evaluate_model, predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_modelinput_data,
                inputs=["train_data_for_price_classification", "parameters"],
                outputs=["X_train_for_price_classification", "y_train_for_price_classification"],
                name="create_train_data_for_classification_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_for_price_classification", "y_train_for_price_classification", "parameters"],
                outputs=["classifier", "classification_eval_results"],
                name="train_model_for_price_classification_node",
            ),
            node(
                func=plot_feature_importance,
                inputs=["classifier", "parameters"],
                outputs=None,
                name="plot_feature_importance_for_price_classification_node",
            ),
            node(
                func=plot_learning_curve,
                inputs=["classification_eval_results", "parameters"],
                outputs=None,
                name="plot_learning_curve_for_price_classification_node",
            ),
            node(
                func=create_modelinput_data,
                inputs=["valid_data_for_price_classification", "parameters"],
                outputs=["X_valid_for_price_classification", "y_valid_for_price_classification"],
                name="create_valid_data_for_price_classification_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier", "X_valid_for_price_classification", "y_valid_for_price_classification", "parameters"],
                outputs=None,
                name="evaluate_model_for_price_classification_node",
            ),
            node(
                func=predict,
                inputs=["classifier", "test_data", "parameters"],
                outputs="highend_proba",
                name="predict_for_price_classification_node",
            ),
        ]
    )
