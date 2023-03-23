import numpy as np
import pandas as pd
from typing import Callable, Dict, List
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

METRICS_TO_CALC = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'auc': roc_auc_score,
    'cm': confusion_matrix
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


def evaluate_model(y: np.array, x: np.array, estimator: BaseEstimator, metrics: Dict = METRICS_TO_CALC) -> Dict:
    output = dict()
    for metric, metric_estimator in METRICS_TO_CALC:
        output[metric] = metric_estimator()
