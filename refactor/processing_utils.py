import numpy as np
import pandas as pd
import re
from typing import Dict, List


AC_STRING = 'air conditioning'
BREAKFAST_STRING = 'breakfast'
ELEVATOR_STRING = 'elevator'
HEATING_STRING = 'heating'
INTERNET_STRING = 'internet'
KITCHEN_STRING = 'kitchen'
TV_STRING = 'tv'
WIFI_STRING = 'wifi'
PREPRO_OUTPUT_SCHEMA = {
    'id': int,
    'neighbourhood_group_cleansed': str,
    'property_type': str,
    'room_type': str,
    'latitude': float,
    'longitude': float,
    'accommodates': int,
    'bathrooms': float,
    'bedrooms': int,
    'beds': int,
    'price': int,
    'category': int,
    'tv': int,
    'internet': int,
    'air_conditioning': int,
    'kitchen': int,
    'heating': int,
    'wifi': int,
    'elevator': int,
    'breakfast': int
}
RELEVANT_FEATURES = [
    'id', 'neighbourhood_group_cleansed', 'property_type', 'room_type',
    'latitude', 'longitude', 'accommodates', 'bathrooms_text', 'bedrooms',
    'beds', 'amenities', 'price'
]


def get_only_relevant_features(
        df: pd.DataFrame,
        relevant_columns: List[str] = RELEVANT_FEATURES
) -> pd.DataFrame:
    output = df[relevant_columns]
    return output


def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
    output = df.dropna(axis=0)
    return output


def parse_property_price(price_feature: str, df: pd.DataFrame) -> pd.DataFrame:
    output = df
    output[price_feature] = output.apply(get_price_from_string, args=[price_feature], axis=1)
    return output


def get_price_from_string(row: pd.DataFrame, price_feature: str) -> float:
    property_price = re.findall(r'\d+(?:\.\d+)?', row[price_feature])
    if len(property_price) == 0:
        return np.NaN
    else:
        return float(property_price[0])


def get_num_of_bathrooms_from_string(row: pd.DataFrame, bathrooms_as_str_feature: str) -> int:
    if pd.isna(row[bathrooms_as_str_feature]):
        return np.NaN
    else:
        bathroom_number = re.findall(r'\d+(?:\.\d+)?', row[bathrooms_as_str_feature])
    if len(bathroom_number) == 0:
        return np.NaN
    elif len(bathroom_number) == 1:
        return float(bathroom_number[0])
    else:
        raise ValueError(f'The string "{row[bathrooms_as_str_feature]}" may contain ambiguous information about the number of bathrooms')


def parse_num_of_bathrooms(
        bathrooms_as_str_feature: str,
        bathrooms_as_float_feature: str,
        df: pd.DataFrame
) -> pd.DataFrame:
    output = df.copy()
    output[bathrooms_as_float_feature] = output.apply(
        get_num_of_bathrooms_from_string,
        args=[bathrooms_as_str_feature],
        axis=1
    )
    return output


def remove_records_below_threshold_price(
    price_feature: str,
    price_threshold: int,
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df[df[price_feature] >= price_threshold]


def create_categorical_price_labels(
        price_feature: str,
        bins: List[int],
        labels: List[int],
        df: pd.DataFrame,
) -> pd.DataFrame:
    df_with_price_category = df
    df_with_price_category['category'] = pd.cut(
        df_with_price_category[price_feature],
        bins=bins,
        labels=labels
    )
    return df_with_price_category


def has_amenity(row: pd.DataFrame, amenities_feature: str, amenity_identifier: str) -> bool:
    amenities_as_list = parse_amenity_field(row=row, amenities_feature=amenities_feature)
    amenities_lowercase = [amenity.lower() for amenity in amenities_as_list]
    return any([amenity_identifier.lower() in amenity for amenity in amenities_lowercase])


def parse_amenity_field(row: pd.DataFrame, amenities_feature: str) -> List:
    if isinstance(row[amenities_feature], str):
        return row[amenities_feature].strip('][').replace('"', '').split(', ')
    elif isinstance(row[amenities_feature], list):
        return row[amenities_feature]
    else:
        raise ValueError('The amenities_feature must contain str or list values')


def has_air_conditioning(row: pd.DataFrame, amenities_feature: str) -> bool:
    return has_amenity(row=row, amenities_feature=amenities_feature, amenity_identifier=AC_STRING)


def has_breakfast(row: pd.DataFrame, amenities_feature: str) -> bool:
    return has_amenity(row=row, amenities_feature=amenities_feature, amenity_identifier=BREAKFAST_STRING)


def has_elevator(row: pd.DataFrame, amenities_feature: str) -> bool:
    return has_amenity(row=row, amenities_feature=amenities_feature, amenity_identifier=ELEVATOR_STRING)


def has_heating(row: pd.DataFrame, amenities_feature: str) -> bool:
    return has_amenity(row=row, amenities_feature=amenities_feature, amenity_identifier=HEATING_STRING)


def has_internet(row: pd.DataFrame, amenities_feature: str) -> bool:
    return has_amenity(row=row, amenities_feature=amenities_feature, amenity_identifier=INTERNET_STRING)


def has_kitchen(row: pd.DataFrame, amenities_feature: str) -> bool:
    return has_amenity(row=row, amenities_feature=amenities_feature, amenity_identifier=KITCHEN_STRING)


def has_tv(row: pd.DataFrame, amenities_feature: str) -> bool:
    return has_amenity(row=row, amenities_feature=amenities_feature, amenity_identifier=TV_STRING)


def has_wifi(row: pd.DataFrame, amenities_feature: str) -> bool:
    return has_amenity(row=row, amenities_feature=amenities_feature, amenity_identifier=WIFI_STRING)


def get_amenties_available(amenities_feature: str, df: pd.DataFrame) -> pd.DataFrame:
    df_with_amenities = df
    amenities_to_get = {
        'air_conditioning': has_air_conditioning,
        'breakfast': has_breakfast,
        'elevator': has_elevator,
        'heating': has_heating,
        'internet': has_internet,
        'kitchen': has_kitchen,
        'tv': has_tv,
        'wifi': has_wifi
    }
    for amenity, amenity_callable in amenities_to_get.items():
        df_with_amenities[amenity] = df_with_amenities.apply(amenity_callable, args=[amenities_feature], axis=1)
    return df_with_amenities.drop(columns=[amenities_feature])


def apply_processing_output_schema(df: pd.DataFrame, schema: Dict = PREPRO_OUTPUT_SCHEMA) -> pd.DataFrame:
    output = df.astype(schema)
    return output[schema.keys()]
