"""
This is a boilerplate pipeline '01_classify_price'
generated using Kedro 0.18.12
"""
from typing import Dict, Tuple, Any
import pandas as pd
import scuc23.modules.lgbm_util as lgbm_util
import scuc23.modules.xgb_util as xgb_util
import scuc23.modules.data_util as data_util

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

import mlflow

def create_modelinput_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    opts=parameters["01_classify_price"]
    return data_util.create_modelinput_data(data, opts)

def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> (Any, Dict):
    """
    モデルを学習させる

    Args:
        X_train: 特徴量
        y_train: 目的変数
        parameters: parameters/01_classify_price.ymlのDictionary

    Returns:
        学習済みモデル
    """
    opts=parameters["01_classify_price"]
    if opts["model"] == "lgbm":
        return lgbm_util.train_lgbm(X_train, y_train, opts["model_options"])
    elif opts["model"] == "xgb":
        return xgb_util.train_xgb(X_train, y_train, opts)
    else:
        raise NotImplementedError(f"{opts['model']} is not implemented yet")    


def plot_feature_importance(regressor, parameters: Dict):
    opts=parameters["01_classify_price"]
    if opts["model"] == "lgbm":
        lgbm_util.save_lgbm_graph(regressor, "price_classificatiton")


def plot_learning_curve(eval_results: Dict, parameters: Dict):
    opts=parameters["01_classify_price"]
    if opts["model"] == "lgbm":
        lgbm_util.save_lgbm_learning_curve(eval_results, "price_classificatiton", ["valid auc-mean"])


import plotly.express as px
import os
from sklearn.metrics import roc_curve, auc
from datetime import datetime
def evaluate_model(classifier, X_valid: pd.DataFrame, y_valid: pd.Series, parameters: Dict):
    """Calculates and logs the coefficient of determination.

    Args:
        classifier: モデル
        X_valid: 評価用特徴量データ(features).
        y_valid: 評価用目的変数（high-end）.
        parameters: parameters/01_classify_price.ymlのDictionary
    """
    y_pred_folds = [booster.predict(X_valid) for booster in classifier.raw_boosters]
    y_pred = pd.DataFrame(y_pred_folds).T.mean(axis=1)
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    start_datetime=mlflow.active_run().data.tags['custom.startDateTime']
    output_dir_base="data/08_reporting/roc_curve"
    output_dir=os.path.join(output_dir_base, start_datetime)
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, "roc_curve.png"))


def predict(classifier, test_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Calculates and logs the coefficient of determination.

    Args:
        classifier: モデル
        test_data: テスト用データ
        parameters: parameters/01_classify_price.ymlのDictionary
    """
    opts=parameters["01_classify_price"]
    if opts["model"] == "lgbm":
        y_pred = [booster.predict(test_data) for booster in classifier.raw_boosters]
    elif opts["model"] == "xgb":
        y_pred = classifier.predict(test_data)
    output_df = pd.DataFrame(y_pred).T.mean(axis=1).to_frame()
    output_df.columns = ['high-end proba']
    output_df.insert(0, 'index', range(27532, 27532 + len(output_df)))
    return output_df
