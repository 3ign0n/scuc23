"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.11
"""
import mlflow
from typing import Dict
import pandas as pd
import category_encoders as ce
from datetime import datetime

def enable_autologging(parameters: Dict):
    mlflow.set_tag("mlflow.runName", datetime.now().isoformat())
    mlflow.autolog()

def do_label_encoding(data: pd.DataFrame) -> pd.DataFrame:
    categorical_column_list = ["region","manufacturer","condition","cylinders","fuel","title_status","transmission","drive","size","type","paint_color","state"]

    ce_oe = ce.OrdinalEncoder(cols=categorical_column_list, handle_unknown='impute')
    enc_data = ce_oe.fit_transform(data)

    #値を1の始まりから0の始まりにする
    for column_name in categorical_column_list:
        enc_data[column_name] = enc_data[column_name] - 1
    
    return enc_data

def preprocess_train_data(train_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for train_data.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data
    """
    return do_label_encoding(train_data)

def preprocess_test_data(test_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for train_data.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data
    """
    return do_label_encoding(test_data)
