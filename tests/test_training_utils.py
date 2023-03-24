from refactor.model_utils import get_model_features, get_model_target
from refactor.pipeline_utils import source_parquet

PROCESSED = source_parquet('./data/processed/processed_diego.parquet')


def test_get_model_features():
    features_to_get = ['price', 'tv', 'internet']
    expected = PROCESSED[features_to_get]
    actual = get_model_features(df=PROCESSED, features=features_to_get)
    assert expected.equals(actual)


def test_get_model_target():
    features_to_get = 'category'
    expected = PROCESSED[features_to_get]
    actual = get_model_target(df=PROCESSED, target=features_to_get)
    assert expected.equals(actual)


def test_evaluation():
    pass
