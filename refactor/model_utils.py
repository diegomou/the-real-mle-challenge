from functools import partial
import numpy as np
import pandas as pd
from typing import Callable, Dict, List
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report

METRICS_TO_CALC = {
    'accuracy': accuracy_score,
    'by_class_rep': partial(classification_report, output_dict=True)
}


def get_model_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df[target]


def get_model_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return df[features]


def fit_model(
    estimator: Callable,
    estimator_args: Dict,
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> BaseEstimator:
    clf = estimator(**estimator_args)
    clf.fit(x, y)
    return clf


def evaluate_model(
    y_true: np.array,
    y_pred: np.array,
    metrics: Dict = METRICS_TO_CALC
) -> Dict:
    output = dict()
    for metric, metric_estimator in metrics.items():
        output[metric] = metric_estimator(y_true, y_pred)
    return output
