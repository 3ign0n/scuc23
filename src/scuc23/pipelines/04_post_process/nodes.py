"""
This is a boilerplate pipeline 'post_process'
generated using Kedro 0.18.12
"""
from typing import Dict
import pandas as pd

def ensemble(y_pred_under_threshold: pd.DataFrame, y_pred_over_threshold: pd.DataFrame, highend_proba: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    output_df = pd.merge(y_pred_under_threshold, y_pred_over_threshold, on='index')
    output_df = pd.merge(output_df, highend_proba, on='index')
    output_df['y_pred'] = output_df['y_pred under threshold'] * (1-output_df['high-end proba']) + output_df['y_pred over threshold'] * output_df['high-end proba']
    return output_df.drop(columns=['y_pred under threshold', 'y_pred over threshold', 'high-end proba'])
