from typing import Dict, List, Any
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

class ModelExtractionCallback(xgb.callback.TrainingCallback):
    """Callback class for retrieving trained model from xgboost.cv()
    """

    def __init__(self):
        self._cvfolds = None

    def after_training(self, model):
        # Saving _CVBooster object.
        if self._cvfolds is None:
            self._cvfolds = model.cvfolds
        return model

    def _assert_called_cb(self):
        if self._cvfolds is None:
            # Throw exception if the callback class is not called.
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        return [cvpack.bst for cvpack in self._cvfolds]
    
    """
    xgb.callback.TrainingCallbackをpickle化したら、読み出し時にエラーになった
    参考: https://blog.amedama.jp/entry/xgboost-cv-model

    def __getattr__(self, name):
        def _wrap(*args, **kwargs):
            ret = []
            for cvpack in self.cvpack:
                # それぞれの Booster から指定されたアトリビュートを取り出す
                attr = getattr(cvpack.bst, name)
                if inspect.ismethod(attr):
                    # オブジェクトがメソッドなら呼び出した結果を入れる
                    res = attr(*args, **kwargs)
                    ret.append(res)
                else:
                    # それ以外ならそのまま入れる
                    ret.append(attr)
            return ret
        return _wrap
    """

class XgbBestModel:
    def __init__(self, cvbest_list: List):
        self._cvbest_list = cvbest_list

    def predict(self, df: pd.DataFrame):
        test_data = xgb.DMatrix(df)

        pred_list = []
        for cvbest in self._cvbest_list:
            pred = cvbest.predict(test_data)
            pred_list.append(pred)

        return pred_list

def xgb_mape(preds, dtrain):
   labels = dtrain.get_label()
   return('mape', np.mean(np.abs((labels - preds) / (labels + 1))))

def train_xgb(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        parameters: Dict
        ) -> Any:

    train_data = xgb.DMatrix(X_train, label=y_train)

    folds = StratifiedKFold(n_splits=parameters["model_options"]["xgb_params"]["num_folds"],
                            shuffle=True,
                            random_state=parameters["random_state"])
    extraction_cb = ModelExtractionCallback()

    result = xgb.cv(
        parameters["model_options"]["xgb_params"]["hyperparams"],
        train_data,
        num_boost_round=parameters["model_options"]["xgb_params"]["num_boost_round"],
        nfold=parameters["model_options"]["xgb_params"]["num_folds"],
        folds=folds,
        feval=xgb_mape,
        maximize=False,
        verbose_eval=parameters["model_options"]["xgb_params"]["verbose_eval"],
        seed=parameters["random_state"],
        callbacks=[
            xgb.callback.EarlyStopping(rounds=parameters["model_options"]["xgb_params"]["early_stopping_rounds"], save_best=False, metric_name=parameters["model_options"]["xgb_params"]["hyperparams"]["eval_metric"]),
            extraction_cb,
        ]
    )
    print(result)

    return XgbBestModel(extraction_cb.boosters_proxy)
