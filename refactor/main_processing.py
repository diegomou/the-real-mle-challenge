from functools import partial
import numpy as np

from pipeline_utils import sink_parquet, source_csv, seq_steps_pipeline, run_pipeline
from processing_utils import (
    get_only_relevant_features, drop_nans, parse_num_of_bathrooms, get_amenties_available,
    remove_records_below_threshold_price, create_categorical_price_labels,
    apply_processing_output_schema, parse_property_price, map_categorical_features
)

PATH_TO_RAW = './data/raw/listings.csv'
PATH_TO_SAVE = './data/processed/processed_diego.parquet'
AMENITIES_FEATURE = 'amenities'
BATHROOMS_STR = 'bathrooms_text'
BATHROOMS_FLOAT = 'bathrooms'
PRICE_FEATURE = 'price'
PRICE_THRESHOLD = 10
PRICE_BINS = [10, 90, 180, 400, np.inf]
PRICE_LABELS = [0, 1, 2, 3]
FEATURES_MAPPINGS = {
    'room_type': {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4},
    'neighbourhood_group_cleansed': {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}
}
CALLABLES_TO_APPLY = [
    get_only_relevant_features,
    partial(parse_num_of_bathrooms, BATHROOMS_STR, BATHROOMS_FLOAT),
    partial(parse_property_price, PRICE_FEATURE),
    partial(remove_records_below_threshold_price, PRICE_FEATURE, PRICE_THRESHOLD),
    partial(create_categorical_price_labels, PRICE_FEATURE, PRICE_BINS, PRICE_LABELS),
    partial(get_amenties_available, AMENITIES_FEATURE),
    partial(parse_num_of_bathrooms, BATHROOMS_STR, BATHROOMS_FLOAT),
    partial(map_categorical_features, FEATURES_MAPPINGS),
    drop_nans,
    apply_processing_output_schema
]
run_pipeline(
    source=partial(source_csv, PATH_TO_RAW),
    sink=partial(sink_parquet, PATH_TO_SAVE),
    pipeline_steps=partial(seq_steps_pipeline, CALLABLES_TO_APPLY)
)
