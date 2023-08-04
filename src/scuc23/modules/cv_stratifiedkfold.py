from typing import Dict, List
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def split_with_stratifiedkfold(X: pd.DataFrame, y: pd.DataFrame, parameters: Dict) -> List:
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameters["random_state"])
    cv = fold.split(X, y)

    # split の返り値は generator なので list 化して何度も iterate できるようにしておく
    return list(cv) 
