import pandas as pd
from typing import Callable, List


def source_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def sink_parquet(path: str, df: pd.DataFrame):
    df.to_parquet(path)


def sink_return(df: pd.DataFrame) -> pd.DataFrame:
    return df


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


def run_pipeline(sink: Callable, source: Callable, pipeline_steps: Callable):
    input = source()
    pipeline_outcome = pipeline_steps(input)
    return sink(pipeline_outcome)
