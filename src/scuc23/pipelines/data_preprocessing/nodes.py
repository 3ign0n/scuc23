"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.11
"""
import mlflow
from typing import Dict
import pandas as pd
import category_encoders as ce
from datetime import datetime
import os
import random
import numpy as np

def set_random_state(parameters: Dict):
    random_state = parameters['random_state']
    os.environ['PYTHONHASHSEED'] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)


def enable_autologging(parameters: Dict):
    mlflow.set_tag("mlflow.runName", datetime.now().isoformat())
    mlflow.autolog()


def __preprocess_column_region_state(data: pd.DataFrame, region_state_data: pd.DataFrame) -> pd.DataFrame:
    tmp_df = pd.merge(data, region_state_data, on='region', how='left')
    tmp_df['state']=tmp_df['state_x'].fillna(tmp_df['state_y'])
    tmp_df = tmp_df.drop(columns=['region', 'state_x', 'state_y'])
    return tmp_df

def __preprocess_column_year(data: pd.DataFrame) -> pd.DataFrame:
    # 2999年以上の値はおかしいので、入力ミスと考え、-1000する
    data.loc[data['year']>=2999, 'year'] = data['year'] - 1000
    return data

def __preprocess_column_manufacturer(data: pd.DataFrame) -> pd.DataFrame:
    data['manufacturer'] = data['manufacturer'].str.normalize('NFKC')
    return data


def __preprocess_column_odometer(data: pd.DataFrame) -> pd.DataFrame:
    # オドメーターが負数というのはおかしいので、入力ミスと考え、正数に直す
    data.loc[data['odometer']<0, 'odometer'] = data['odometer'] * -1
    return data


def __preprocess_column_size(data: pd.DataFrame) -> pd.DataFrame:
    data['size'] = data['size'].str.replace('ー', '-')
    data['size'] = data['size'].str.replace('−', '-')    
    return data


def __apply_preprocessing_rules(data: pd.DataFrame, region_state_data: pd.DataFrame) -> pd.DataFrame:
    tmp_df = __preprocess_column_region_state(data, region_state_data)
    tmp_df = __preprocess_column_year(tmp_df)
    tmp_df = __preprocess_column_manufacturer(tmp_df)
    tmp_df = __preprocess_column_odometer(tmp_df)
    return __preprocess_column_size(tmp_df)


def preprocess_train_data(train_data: pd.DataFrame, region_state_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for train_data.

    Args:
        train_data: Raw data.
    Returns:
        Preprocessed data
    """
    return __apply_preprocessing_rules(train_data, region_state_data)


def preprocess_test_data(test_data: pd.DataFrame, region_state_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for train_data.

    Args:
        test_data: Raw data.
    Returns:
        Preprocessed data
    """
    return __apply_preprocessing_rules(test_data, region_state_data)


from pandas_profiling import ProfileReport
def save_pandas_profiling(train_data: pd.DataFrame, test_data: pd.DataFrame):
    OUTPUT_DIR_BASE="data/08_reporting/pandas_profiling"
    OUTPUT_DIR=os.path.join(OUTPUT_DIR_BASE, datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%fZ"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    profile = ProfileReport(train_data, title="Pandas Profiling Report(train data)")
    profile.to_file(os.path.join(OUTPUT_DIR, "train.html"))
    profile = ProfileReport(test_data, title="Pandas Profiling Report(test data)")
    profile.to_file(os.path.join(OUTPUT_DIR, "test.html"))


def preprocess_do_label_encoding(data: pd.DataFrame) -> pd.DataFrame:
    categorical_column_list = ["manufacturer","condition","cylinders","fuel","title_status","transmission","drive","size","type","paint_color","state"]

    ce_oe = ce.OrdinalEncoder(cols=categorical_column_list, handle_unknown='impute')
    enc_data = ce_oe.fit_transform(data)

    # 値を1の始まりから0の始まりにする
    for column_name in categorical_column_list:
        enc_data[column_name] = enc_data[column_name] - 1
    
    return enc_data

