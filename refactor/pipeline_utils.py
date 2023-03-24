import json
import pandas as pd
import pickle
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from typing import Callable, Dict, List

from model_utils import (
    get_model_features, get_model_target, fit_model, evaluate_model
)

def source_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def source_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def sink_parquet(path: str, df: pd.DataFrame):
    df.to_parquet(path)


def sink_json(path: str, payload: Dict):
    with open(path, 'w') as file:
        json.dump(payload, file)


def sink_return(df: pd.DataFrame) -> pd.DataFrame:
    return df


def sink_model(path_to_save: str, model: BaseEstimator):
    with open(path_to_save, 'wb') as file:
        pickle.dump(model, file)


def seq_steps_pipeline(
        callables_to_apply: List[Callable],
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    To create a pipeline callable that runs sequential steps. As this 
    function applies the callables in callables_to_apply following the
    list order, the user must be careful when you define that order.
    """
    outcome = df.copy()
    for callable in callables_to_apply:
        outcome = callable(outcome)
    return outcome


def train_model_pipeline(
    source: Callable,
    sink_model: Callable,
    sink_eval: Callable,
    model_specs: Dict
):
    processed = source()
    features = get_model_features(df=processed, features=model_specs['features'])
    target = get_model_target(df=processed, target=model_specs['target'])
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        **model_specs['train_test_split_params']
    )
    model = fit_model(
        estimator=model_specs['estimator'],
        estimator_args=model_specs['estimator_params'],
        x=X_train,
        y=y_train
    )
    metrics_eval = evaluate_model(
        y_true=y_test,
        y_pred=model.predict(X_test),
        y_pred_proba=model.predict_proba(X_test)
    )
    return sink_model(model), sink_eval(metrics_eval)


def run_pipeline(sink: Callable, source: Callable, pipeline_steps: Callable):
    input = source()
    pipeline_outcome = pipeline_steps(input)
    return sink(pipeline_outcome)
