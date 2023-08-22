from typing import Dict, Any
import lightgbm as lgbm
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
        ) -> (Any, Dict):

    # get_dummiesしたときは指定したらダメっぽい
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    #category_columns=X_train.columns.drop(['year', 'odometer', 'cylinders'])

    train_data = lgbm.Dataset(X_train, label=y_train)

    extraction_cb = ModelExtractionCallback()

    result = lgbm.cv(
        parameters["lgbm_params"]["hyperparams"],
        train_data,
        num_boost_round=parameters["lgbm_params"]["num_boost_round"],
        nfold=parameters["lgbm_params"]["num_folds"],
        stratified=parameters["lgbm_params"]["is_stratified"],
        shuffle=parameters["lgbm_params"]["is_shuffle"],
        #categorical_feature=category_columns,
        callbacks=[
            lgbm.early_stopping(stopping_rounds=parameters["lgbm_params"]["early_stopping_rounds"], verbose=True),
            lgbm.log_evaluation(parameters["lgbm_params"]["verbose_eval"]),
            extraction_cb,
        ]
    )
    #print(result)

    return extraction_cb, result

import os
from datetime import datetime
import mlflow
import matplotlib.pyplot as plt
def save_lgbm_graph(model, id: str):
    start_datetime=mlflow.active_run().data.tags['custom.startDateTime']
    output_dir_base=f"data/08_reporting/feature_importance_{id}"
    output_dir=os.path.join(output_dir_base, start_datetime)
    os.makedirs(output_dir, exist_ok=True)

    for i, booster in enumerate(model.raw_boosters):
        lgbm.plot_importance(booster,height=0.5,figsize=(8,16))
        plt.savefig(os.path.join(output_dir, f"importance_{i}.png"))
    plt.close()

    output_dir_base=f"data/08_reporting/decision_tree_{id}"
    output_dir=os.path.join(output_dir_base, start_datetime)
    os.makedirs(output_dir, exist_ok=True)

    # https://stackoverflow.com/questions/61894279/writing-create-tree-digraph-plot-to-a-png-file-in-python
    for i, booster in enumerate(model.raw_boosters):
        viz=lgbm.create_tree_digraph(booster)
        viz.render(format='png', filename=os.path.join(output_dir, f"tree_digraph_{i}.dot"))
        #graph = pydotplus.graph_from_dot_data(viz.source)
        #graph.write_png(os.path.join(output_dir, f"tree_digraph_{i}.png"))


def save_lgbm_learning_curve(eval_results: Dict, id: str, metrics: [str]):
    # 3つ以上metricsがあることはいまは考慮してない
    plot_df = pd.DataFrame({'iterations': [i for i, _ in enumerate(eval_results[metrics[0]], start=1)], f"performance({metrics[0]})": eval_results[metrics[0]]})
    if len(metrics) == 2:
        plot_df[f"performance({metrics[1]})"]=eval_results[metrics[1]]

    fig= make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=plot_df["iterations"], y=plot_df[f"performance({metrics[0]})"], name=metrics[0], line=dict(color="blue")))
    if len(metrics) == 2:
        fig.add_trace(go.Scatter(x=plot_df["iterations"], y=plot_df[f"performance({metrics[1]})"], name=metrics[1], line=dict(color="purple")),secondary_y=True)
    fig.update_layout(
        title_text='performance', title_x=0.5
    )
    fig.update_xaxes(title_text='iterations')
    fig.update_yaxes(title_text=metrics[0], secondary_y=False)
    if len(metrics) == 2:
        fig.update_yaxes(title_text=metrics[1], secondary_y=True)

    start_datetime=mlflow.active_run().data.tags['custom.startDateTime']
    output_dir_base=f"data/08_reporting/iteration_performance_{id}"
    output_dir=os.path.join(output_dir_base, start_datetime)
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, f"iteration_performance_{id}.png"))
