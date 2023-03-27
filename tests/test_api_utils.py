import numpy as np
import pytest
from sklearn.base import BaseEstimator

from model_api.api_utils import (
    get_model_features, map_values, make_model_prediction,
    make_predictions_airbnb
)

EXAMPLE_ALL = {
    "id": 1001,
    "accommodates": 4,
    "room_type": "Entire home/apt",
    "beds": 2,
    "bedrooms": 1,
    "bathrooms": 2,
    "neighbourhood": "Brooklyn",
    "tv": 1,
    "elevator": 1,
    "internet": 0,
    "latitude": 40.71383,
    "longitude": -73.9658
}
EXAMPLE_NONE = {
    "id": 1001,
    "accommodates": 4,
    "room_type": "Entire home/apt",
    "beds": None,
    "bedrooms": 1,
    "bathrooms": 2,
    "neighbourhood": "Brooklyn",
    "tv": 1,
    "elevator": 1,
    "internet": 0,
    "latitude": 40.71383,
    "longitude": -73.9658
}
MAPPING = {
    'dog': 0,
    'cat': 1,
    'parrot': 2
}


class MockEstimator(BaseEstimator):
    def predict(instance: np.array) -> int:
        return np.array([1]) if np.array_equal(instance, np.array([1, 0, 1])) else np.array([0])


def test_get_model_features_ok():
    actual = {
        "id": 1001,
        "beds": 2
    }
    expected = get_model_features(request=EXAMPLE_ALL, model_features=['id', 'beds'])
    assert actual == expected


def test_get_model_features_missing_feature():
    with pytest.raises(
        KeyError, match='The feature random is not available in the payload sent'
    ):
        get_model_features(request=EXAMPLE_ALL, model_features=['id', 'beds', 'random'])


def test_get_model_features_missing_none_feature():
    with pytest.raises(
        ValueError, match='The feature beds has null values'
    ):
        get_model_features(request=EXAMPLE_NONE, model_features=['id', 'beds'])


def test_map_values_value_in_mapping():
    example = 'cat'
    expected = 1
    actual = map_values(value_to_map=example, mapping=MAPPING)
    assert actual == expected


def test_map_values_value_not_in_mapping():
    example = 'horse'
    with pytest.raises(
        KeyError, match='horse wasn`t found in the mapping dict'
    ):
        map_values(value_to_map=example, mapping=MAPPING)


def test_make_model_prediction_label_1():
    example = np.array([1, 0, 1])
    actual = make_model_prediction(model=MockEstimator, features=example)
    expected = 1
    assert actual == expected


def test_make_model_prediction_label_0():
    example = np.array([1, 0, 0])
    actual = make_model_prediction(model=MockEstimator, features=example)
    expected = 0
    assert actual == expected


def test_make_predictions_airbnb():
    example = np.array([1, 0, 0])
    actual = make_predictions_airbnb(model=MockEstimator, features=example)
    expected = 'low'
    assert actual == expected
