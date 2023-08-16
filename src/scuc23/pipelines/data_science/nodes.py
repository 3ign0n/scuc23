"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.11
"""
from typing import Dict, Tuple, Any
import pandas as pd
import scuc23.modules.lgbm_util as lgbm_util
import scuc23.modules.xgb_util as xgb_util

from sklearn.metrics import r2_score
import logging

import mlflow

def create_train_data(train_data: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    学習用データを説明変数と目的変数に分割する

    Args:
        train_data: 学習用データ.
        parameters: parameters/data_science.ymlのDictionary
    Returns:
        学習用data.
    """
    opts = parameters["model_options"]
    X_train = train_data[opts["features"]]
    y_train = train_data[opts["y_label"]]
    return X_train, y_train

def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> Any:
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


#import plotly.express as px
#import os
#from datetime import datetime
#def evaluate_model(regressor, X_test: pd.DataFrame, y_test: pd.Series, parameters: Dict):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: モデル
        X_test: 評価データ.
        y_test: 評価結果（price）.
    """
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)


    # 予測と価格の値を散布図に
    df = pd.DataFrame(y_test).rename(columns={'price': 'y_test'}).reset_index()
    df['y_pred'] = pd.Series(y_pred)
    fig = px.scatter(df, x='y_test', y='y_pred', title='y_test vs y_pred', trendline='ols', trendline_color_override='red')

    start_datetime=mlflow.active_run().data.tags['custom.startDateTime']
    output_dir_base="data/08_reporting/scatterplot_pred_vs_valid"
    output_dir=os.path.join(output_dir_base, start_datetime)
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, "scatterplot_pred_vs_valid.png"))
    """

def plot_feature_importance(regressor, parameters: Dict):
    if parameters["model"] == "lgbm":
        lgbm_util.save_lgbm_graph(regressor)

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

