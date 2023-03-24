from functools import partial
from sklearn.ensemble import RandomForestClassifier

from pipeline_utils import source_parquet, sink_model, train_model_pipeline, sink_json

PATH_TO_PROC = './data/processed/processed_diego.parquet'
PATH_TO_MODEL = './models/model_diego.pkl'
PATH_TO_METRICS = './models/training_evaluation/metrics_diego.json'
MODEL_SPECS = {
    'estimator': RandomForestClassifier,
    'estimator_params': {
        'n_estimators': 500,
        'random_state': 0,
        'class_weight': 'balanced',
        'n_jobs': 4
    },
    'features': [
        'neighbourhood_group_cleansed', 'room_type', 'accommodates', 'bathrooms', 'bedrooms'
    ],
    'target': 'category',
    'train_test_split_params': {
        'test_size': 0.15,
        'random_state': 1
    }
}
train_model_pipeline(
    source=partial(source_parquet, PATH_TO_PROC),
    sink_model=partial(sink_model, PATH_TO_MODEL),
    sink_eval=partial(sink_json, PATH_TO_METRICS),
    model_specs=MODEL_SPECS
)
