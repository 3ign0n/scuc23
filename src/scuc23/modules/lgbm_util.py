from typing import Dict, Any
import lightgbm as lgbm
import pandas as pd

def train_lgbm(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        parameters: Dict
        ) -> Any:

    train_data = lgbm.Dataset(X_train, label=y_train)
    validation_data = lgbm.Dataset(X_validation, label=y_validation)

    return lgbm.train(
        parameters["model_options"]["lgbm_params"]["hyperparams"],
        train_data,
        valid_sets=[validation_data],
        num_boost_round=parameters["model_options"]["lgbm_params"]["num_boost_round"],
        callbacks=[
            lgbm.early_stopping(stopping_rounds=parameters["model_options"]["lgbm_params"]["early_stopping_rounds"], verbose=True),
            lgbm.log_evaluation(parameters["model_options"]["lgbm_params"]["verbose_eval"])
        ]
    )
