import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator
from typing import Dict, List, Union

AIRBNB_FEAT_MAPPING = {
    'room_type': {
        "Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4
    },
    'neighbourhood':
        {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}
}
AIRBNB_OUTPUT_MAPPING = {
    0: "low",
    1: "mid",
    2: "high",
    3: "lux"
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


def map_values(value_to_map: Union[str, int], mapping: Dict):
    """
    Maps a string by using a mapping dictionary. The keys are
    the values of a given feature, and the values are the mapping
    values for that feature.
    If the mapper finds a match between value_to_map and one key
    in mapping dict, then returns the mapped value. If not, it
    returns the original value.
    """
    if value_to_map in mapping.keys():
        return mapping[value_to_map]
    else:
        raise KeyError(f'{value_to_map} wasn`t found in the mapping dict')


def make_model_prediction(model: BaseEstimator, features: np.array) -> int:
    prediction_raw = model.predict(features)
    return int(prediction_raw[0])


def process_features_airbnb(features: Dict) -> np.array:
    """
    Giving a dict with the features given to the API, this function
    maps the string & categorical features into their corresponding
    integer values. Those values were assigned by the DS who trained
    the model.
    """
    for feature, mapping in AIRBNB_FEAT_MAPPING.items():
        features[feature] = map_values(features[feature], mapping)
    return np.fromiter(features.values(), dtype=float).reshape(1, -1)


def make_predictions_airbnb(model: BaseEstimator, features: np.array) -> str:
    prediction_int = make_model_prediction(model=model, features=features)
    return map_values(value_to_map=prediction_int, mapping=AIRBNB_OUTPUT_MAPPING)
