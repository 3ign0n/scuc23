"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.11
"""
from typing import Dict, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import scuc23.modules.lgbm_util as lgbm_util

from sklearn.metrics import r2_score
import logging

def create_train_test_data(train_data: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    学習用データを分割する

    Args:
        train_data: 学習用データ.
        parameters: parameters/data_science.ymlのDictionary
    Returns:
        Split data.
    """
    opts = parameters["model_options"]
    X = train_data[opts["features"]]
    y = train_data[opts["y_label"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, parameters: Dict) -> Any:
    """
    モデルを学習させる

    Args:
        X_train: 特徴量
        y_train: 目的変数

    Returns:
        学習モデル
    """
    if parameters["model"] == "lgbm":
        return lgbm_util.train_lgbm(X_train, y_train, X_test, y_test, parameters)
    else:
        raise NotImplementedError(f"{parameters['model']} is not implemented yet")    


def evaluate_model(regressor, X_test: pd.DataFrame, y_test: pd.Series, parameters: Dict):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: モデル
        X_test: 評価データ.
        y_test: 評価結果（price）.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)


def predict(regressor, test_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: モデル
        train_data: 学習用データ
        parameters: parameters/data_science.ymlのDictionary
    """
    y_pred = regressor.predict(test_data)
    return pd.DataFrame(y_pred)
