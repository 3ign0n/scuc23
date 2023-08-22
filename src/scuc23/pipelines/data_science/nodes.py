"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.11
"""
from typing import Dict, Tuple, Any
import pandas as pd
import scuc23.modules.lgbm_util as lgbm_util
import scuc23.modules.xgb_util as xgb_util

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import logging

import mlflow

def create_modelinput_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    学習用データを説明変数と目的変数に分割する

    Args:
        train_data: 学習用データ.
        parameters: parameters/data_science.ymlのDictionary
    Returns:
        学習用data.
    """
    opts = parameters["valid_params"]
    y = data[opts["y_label"]]
    X = data.drop(columns=[opts["y_label"]])
    return X, y

def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> (Any, Dict):
    """
    モデルを学習させる

    Args:
        X_train: 特徴量
        y_train: 目的変数

    Returns:
        学習モデル
    """
    if parameters["model"] == "lgbm":
        return lgbm_util.train_lgbm(X_train, y_train, parameters)
    elif parameters["model"] == "xgb":
        return xgb_util.train_xgb(X_train, y_train, parameters)
    else:
        raise NotImplementedError(f"{parameters['model']} is not implemented yet")    


import plotly.express as px
import os
from datetime import datetime
def evaluate_model(regressor, X_valid: pd.DataFrame, y_valid: pd.Series, parameters: Dict):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: モデル
        X_valid: 評価データ.
        y_valid: 評価結果（price）.
    """
    y_pred_folds = regressor.boosters_proxy.predict(X_valid)
    y_pred = pd.DataFrame(y_pred_folds).T.mean(axis=1)
    score = r2_score(y_valid, y_pred)
    mape = mean_absolute_percentage_error(y_valid, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2:%.6f, mape:%.6f on test data.", score, mape)


    # 予測と価格の値を散布図に
    df = pd.DataFrame(y_valid).rename(columns={'price': 'y_valid'}).reset_index()
    df['y_pred'] = pd.Series(y_pred)
    fig = px.scatter(df, x='y_valid', y='y_pred', title='y_valid vs y_pred', trendline='ols', trendline_color_override='red')

    start_datetime=mlflow.active_run().data.tags['custom.startDateTime']
    output_dir_base="data/08_reporting/scatterplot_pred_vs_valid"
    output_dir=os.path.join(output_dir_base, start_datetime)
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, "scatterplot_pred_vs_valid.png"))


def plot_feature_importance(regressor, parameters: Dict):
    if parameters["model"] == "lgbm":
        lgbm_util.save_lgbm_graph(regressor)


def plot_learning_curve(eval_results: Dict, parameters: Dict):
    if parameters["model"] == "lgbm":
        lgbm_util.save_lgbm_learning_curve(eval_results)


def predict(regressor, test_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: モデル
        train_data: 学習用データ
        parameters: parameters/data_science.ymlのDictionary
    """
    if parameters["model"] == "lgbm":
        y_pred = regressor.boosters_proxy.predict(test_data)
    elif parameters["model"] == "xgb":
        y_pred = regressor.predict(test_data)
    output_df = pd.DataFrame(y_pred).T.mean(axis=1).to_frame()
    output_df.insert(0, 'index', range(27532, 27532 + len(output_df)))
    return output_df


def post_process(data: pd.DataFrame):
    _  = mlflow.last_active_run()

