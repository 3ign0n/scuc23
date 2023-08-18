"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.11
"""
import mlflow
from typing import Dict
import pandas as pd
import category_encoders as ce
from datetime import datetime, timedelta, timezone
import os
import random
import numpy as np

def set_random_state(parameters: Dict):
    random_state = parameters['random_state']
    os.environ['PYTHONHASHSEED'] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)


def enable_autologging(parameters: Dict):
    now=datetime.now()

    # https://stackoverflow.com/questions/57199472/is-it-possible-to-set-change-mlflow-run-name-after-run-initial-creation
    # これがどこに反映されるのか不明。mlflow.active_run().info.run_nameには反映されてない
    mlflow.set_tag("mlflow.runName", now.isoformat()) 

    # あとで今回の実行時間を使ってディレクトリ作成したいので、active_runのtagとして保持しておく
    mlflow.active_run().data.tags['custom.startDateTime']=now.strftime("%Y-%m-%dT%H.%M.%S.%fZ")
    mlflow.autolog()


def __preprocess_column_year(data: pd.DataFrame) -> pd.DataFrame:
    # 2999年以上の値はおかしいので、入力ミスと考え、-1000する
    data.loc[data['year']>=2999, 'year'] = data['year'] - 1000
    return data


def __preprocess_column_manufacturer(data: pd.DataFrame) -> pd.DataFrame:
    data['manufacturer'] = data['manufacturer'].str.lower()
    data['manufacturer'] = data['manufacturer'].str.normalize('NFKC')
    
    # nissanにunicodeの変な文字が混じっていた
    data['manufacturer'] = data['manufacturer'].str.replace(u"\u0455", 's')
    # toyotaにunicodeの変な文字が混じっていた
    data['manufacturer'] = data['manufacturer'].str.replace(u"\u0430", 'a')
    # subaru, saturnにunicodeの変な文字が混じっていた
    data['manufacturer'] = data['manufacturer'].str.replace(u"\u03b1", 'a')
    # volkswagenにunicodeの変な文字が混じっていた
    data['manufacturer'] = data['manufacturer'].str.replace(u"\u043e", 'o')
    # chryslerにunicodeの変な文字が混じっていた
    data['manufacturer'] = data['manufacturer'].str.replace(u"\u1d04", 'c')
    return data


def __preprocess_column_condition(data: pd.DataFrame) -> pd.DataFrame:
    # newとlike newがあるが、like newに寄せる。
    # newのodometerやyearの値見た感じ、新車ってことはありえないので
    data.loc[data['condition']=='new', 'condition'] = 'like new'
    return data


def __preprocess_column_cylinders(data: pd.DataFrame) -> pd.DataFrame:
    # 数値化する
    # otherという文字列もあるが、NaN扱いとなる
    data['cylinders'] = data['cylinders'].str.extract(r'(\d+)').astype(float)
    return data


def __preprocess_column_odometer(data: pd.DataFrame) -> pd.DataFrame:
    # オドメーターが負数というのはおかしいので、入力ミスと考え、正数に直す
    data.loc[data['odometer']<0, 'odometer'] = data['odometer'] * -1

    # 10万キロ超えたら買い替え検討、20万キロ超えたらだいぶやばいと言われているのに走行距離50万キロ超えは明らかな入力ミスと考え、一桁減らす
    data.loc[data['odometer']>500000, 'odometer'] = data['odometer'] / 10
    return data


def __preprocess_column_size(data: pd.DataFrame) -> pd.DataFrame:
    data['size'] = data['size'].str.replace('ー', '-')
    data['size'] = data['size'].str.replace('−', '-')    
    return data


def __preprocess_column_state(data: pd.DataFrame) -> pd.DataFrame:
    """
    regionと深い関係があり、多重共線性を生むのでドロップ
    """
    return data.drop(columns=['state'])


def __apply_preprocessing_rules(data: pd.DataFrame) -> pd.DataFrame:
    tmp_df = __preprocess_column_year(data)
    tmp_df = __preprocess_column_manufacturer(tmp_df)
    tmp_df = __preprocess_column_condition(tmp_df)
    tmp_df = __preprocess_column_cylinders(tmp_df)
    tmp_df = __preprocess_column_odometer(tmp_df)
    tmp_df = __preprocess_column_size(tmp_df)
    return __preprocess_column_state(tmp_df)


def preprocess_train_data(train_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for train_data.

    Args:
        train_data: Raw data.
    Returns:
        Preprocessed data
    """
    return __apply_preprocessing_rules(train_data)


def preprocess_test_data(test_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for train_data.

    Args:
        test_data: Raw data.
    Returns:
        Preprocessed data
    """
    return __apply_preprocessing_rules(test_data)


import mlflow
from pandas_profiling import ProfileReport
def save_pandas_profiling(train_data: pd.DataFrame, test_data: pd.DataFrame):
    start_datetime=mlflow.active_run().data.tags['custom.startDateTime']
    output_dir_base="data/08_reporting/pandas_profiling"
    output_dir=os.path.join(output_dir_base, start_datetime)
    os.makedirs(output_dir, exist_ok=True)

    profile = ProfileReport(train_data, title="Pandas Profiling Report(train data)")
    profile.to_file(os.path.join(output_dir, "train.html"))
    profile = ProfileReport(test_data, title="Pandas Profiling Report(test data)")
    profile.to_file(os.path.join(output_dir, "test.html"))


def preprocess_do_label_encoding(data: pd.DataFrame) -> pd.DataFrame:
    categorical_column_list = ["region","manufacturer","condition","fuel","title_status","transmission","drive","size","type","paint_color"]

    ce_oe = ce.OrdinalEncoder(cols=categorical_column_list, handle_unknown='impute')
    enc_data = ce_oe.fit_transform(data)

    # 値を1の始まりから0の始まりにする
    for column_name in categorical_column_list:
        enc_data[column_name] = enc_data[column_name] - 1
    
    return enc_data

