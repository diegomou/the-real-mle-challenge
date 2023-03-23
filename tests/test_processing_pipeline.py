from functools import partial
import numpy as np
import pandas as pd
from refactor.pipeline_utils import source_csv, sink_return, seq_steps_pipeline, run_pipeline
from refactor.processing_utils import (
    get_only_relevant_features, drop_nans, parse_num_of_bathrooms, get_amenties_available,
    remove_records_below_threshold_price, create_categorical_price_labels,
    apply_processing_output_schema, parse_property_price
)

PATH_TO_RAW = './tests/test_data/raw_sample.csv'
PATH_TO_PROC = './tests/test_data/processed_sample.parquet'
AMENITIES_FEATURE = 'amenities'
BATHROOMS_STR = 'bathrooms_text'
BATHROOMS_FLOAT = 'bathrooms'
PRICE_FEATURE = 'price'
PRICE_THRESHOLD = 10
PRICE_BINS = [10, 90, 180, 400, np.inf]
PRICE_LABELS = [0, 1, 2, 3]
CALLABLES_TO_APPLY = [
    get_only_relevant_features,
    partial(parse_num_of_bathrooms, BATHROOMS_STR, BATHROOMS_FLOAT),
    partial(parse_property_price, PRICE_FEATURE),
    partial(remove_records_below_threshold_price, PRICE_FEATURE, PRICE_THRESHOLD),
    partial(create_categorical_price_labels, PRICE_FEATURE, PRICE_BINS, PRICE_LABELS),
    partial(get_amenties_available, AMENITIES_FEATURE),
    partial(parse_num_of_bathrooms, BATHROOMS_STR, BATHROOMS_FLOAT),
    drop_nans,
    apply_processing_output_schema
]


def test_processing_pipeline():
    actual = run_pipeline(
        sink=sink_return,
        source=partial(source_csv, PATH_TO_RAW),
        pipeline_steps=partial(seq_steps_pipeline, CALLABLES_TO_APPLY)
    )
    print(actual)
    expected = pd.read_parquet(PATH_TO_PROC)
    actual = actual.reset_index(drop=True).sort_values(by='id')
    expected = expected.reset_index(drop=True).sort_values(by='id')
    assert len(expected.columns) == len(actual.columns)
    assert all(expected.columns == actual.columns)
    assert expected.equals(actual)
