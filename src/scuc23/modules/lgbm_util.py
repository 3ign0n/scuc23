from typing import Dict, Any
import lightgbm as lgbm
import pandas as pd


class ModelExtractionCallback(object):
    """Callback class for retrieving trained model from lightgbm.cv()
    NOTE: This class depends on '_CVBooster' which is hidden class, so it might doesn't work if the specification is changed.
    NOTE: see the following url in detail https://www.kaggle.com/code/kenmatsu4/using-trained-booster-from-lightgbm-cv-w-callback
    """

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # Saving _CVBooster object.
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            # Throw exception if the callback class is not called.
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        # return Booster object
        return self._model

    @property
    def raw_boosters(self):
        self._assert_called_cb()
        # return list of Booster
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        # return boosting round when early stopping.
        return self._model.best_iteration


def train_lgbm(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        parameters: Dict
        ) -> Any:

    train_data = lgbm.Dataset(X_train, label=y_train)

    extraction_cb = ModelExtractionCallback()

    result = lgbm.cv(
        parameters["model_options"]["lgbm_params"]["hyperparams"],
        train_data,
        num_boost_round=parameters["model_options"]["lgbm_params"]["num_boost_round"],
        nfold=parameters["model_options"]["lgbm_params"]["num_folds"],
        stratified=parameters["model_options"]["lgbm_params"]["is_stratified"],
        shuffle=parameters["model_options"]["lgbm_params"]["is_shuffle"],
        callbacks=[
            lgbm.early_stopping(stopping_rounds=parameters["model_options"]["lgbm_params"]["early_stopping_rounds"], verbose=True),
            lgbm.log_evaluation(parameters["model_options"]["lgbm_params"]["verbose_eval"]),
            extraction_cb,
        ]
    )
    print(result)

    return extraction_cb
