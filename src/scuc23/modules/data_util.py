from typing import Dict, Tuple
import pandas as pd
import scuc23.modules.lgbm_util as lgbm_util


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


def plot_feature_importance(regressor, parameters: Dict):
    if parameters["model"] == "lgbm":
        lgbm_util.save_lgbm_graph(regressor)


def plot_learning_curve(eval_results: Dict, parameters: Dict):
    if parameters["model"] == "lgbm":
        lgbm_util.save_lgbm_learning_curve(eval_results)
