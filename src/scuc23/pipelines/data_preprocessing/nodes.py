"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.11
"""
import pandas as pd

def preprocess_train_data(train_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for train_data.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data
    """
    return train_data

def preprocess_test_data(test_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for train_data.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data
    """
    return test_data
