import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator
from typing import Dict, List

AIRBNB_MAPPING = {
   'room_type': {
        "Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4
    },
    'neighbourhood': {
        "Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5
    }
}


def load_model(path: str) -> BaseEstimator:
    return pickle.load(open(path, 'rb'))


def get_model_features(request: Dict, model_features: List[str]) -> Dict:
    output = dict()
    for feature in model_features:
        try:
            feature_value = request[feature]
        except:
            raise KeyError(f'The feature {feature} is not available in the payload sent')
        if pd.isna(feature_value):
            raise ValueError(f'The feature {feature} has null values')
        output[feature] = feature_value
    return output


def map_values(value_to_map: str, mapping: Dict):
    """
    Maps a string by using a mapping dictionary. The keys are
    the values of a given feature, and the values are the mapping
    values for that feature.
    If the mapper finds a match between value_to_map and one key
    in mapping dict, then returns the mapped value. If not, it
    returns the original value.
    """
    mapped_value = None
    for candidate in mapping.keys():
        if value_to_map == candidate:
            mapped_value = mapping[candidate]
            break
    if mapped_value is None:
        raise KeyError(f'{value_to_map} wasn`t found in the mapping dict')
    else:
        return mapped_value


def make_model_prediction(model: BaseEstimator, features: np.array) -> int:
    prediction_raw = model.predict(features)
    return int(prediction_raw[0])


def airbnb_api_features_processing(features: Dict) -> np.array:
    for feature, mapping in AIRBNB_MAPPING.items():
        features[feature] = map_values(features[feature], mapping)
    return np.fromiter(features.values(), dtype=float).reshape(1, -1)
