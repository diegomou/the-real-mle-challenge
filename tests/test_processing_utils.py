import numpy as np
import pandas as pd
import pytest

from refactor.processing_utils import (
    get_num_of_bathrooms_from_string, parse_num_of_bathrooms, parse_amenity_field,
    has_amenity, has_air_conditioning, has_breakfast, has_heating, has_kitchen,
    has_internet, has_tv, has_wifi, get_amenties_available, get_price_from_string,
    parse_property_price, remove_records_below_threshold_price,
    create_categorical_price_labels, map_categorical_features
)

AMENITIES_EXAMPLE = pd.DataFrame({
        'amenities': [
            '["Kitchen", "Cable TV", "Washer", "Elevator", "Wifi", "Gym", "Heating", "Dryer", "Air conditioning"]'
        ]
    })
PRICES_AS_STRING = pd.DataFrame(
    {
        'price': ['100', '200', '350.27', 'not a price']
    }
)
PRICING_EXAMPLE = pd.DataFrame(
    {
        'price': [12, 9, 8, 23, 99, 50, 22, 34]
    }
)


def test_get_price_from_string():
    actual = PRICES_AS_STRING.apply(
        get_price_from_string,
        args=['price'],
        axis=1
    )
    expected = pd.Series([100, 200, 350.27, np.NaN])
    assert expected.equals(actual)


def test_parse_property_price():
    actual = parse_property_price(price_feature='price', df=PRICES_AS_STRING)
    expected = pd.DataFrame({
        'price': [100, 200, 350.27, np.NaN]
    })
    assert expected.equals(actual)


def test_get_num_of_bath_from_string():
    examples = pd.DataFrame({
        'bathroom_text': ['1 bath', '1.5 baths', '1 bath', '1 shared bath', '5 baths']
    })
    actual = examples.apply(get_num_of_bathrooms_from_string, args=['bathroom_text'], axis=1)
    expected = pd.Series([1, 1.5, 1, 1, 5])
    assert expected.equals(actual)


def test_get_num_of_bath_from_string_empty():
    examples = pd.DataFrame({
        'bathroom_text': [None, np.NaN, pd.NA]
    })
    actual = examples.apply(get_num_of_bathrooms_from_string, args=['bathroom_text'], axis=1)
    expected = pd.Series([np.NaN]*3)
    assert expected.equals(actual)


def test_get_num_of_bath_from_string_string_wout_number():
    examples = pd.DataFrame({
        'bathroom_text': ['some random string', 'Entire home/apt', 'Half-bath']
    })
    actual = examples.apply(get_num_of_bathrooms_from_string, args=['bathroom_text'], axis=1)
    expected = pd.Series([np.NaN]*3)
    assert expected.equals(actual)


def test_get_num_of_bath_from_string_ambiguous():
    example = pd.DataFrame({
        'bathroom_text': ['it may be 1 or 2 bathrooms']
    })
    with pytest.raises(
        ValueError, match='The string "it may be 1 or 2 bathrooms" may contain ambiguous information about the number of bathrooms'
    ):
        example.apply(get_num_of_bathrooms_from_string, args=['bathroom_text'], axis=1)


def test_parse_num_of_bath():
    example = pd.DataFrame({
        'bathroom_text': ['1 bath', '1.5 baths', '1 bath', '1 shared bath', '5 baths']
    })
    actual = parse_num_of_bathrooms(
        bathrooms_as_float_feature='bathrooms',
        bathrooms_as_str_feature='bathroom_text',
        df=example
    )
    example.insert(1, 'bathrooms', [1, 1.5, 1, 1, 5])
    assert example.equals(actual)


def test_parse_amenity_field_list():
    example = pd.DataFrame({
        'amenities': [["some random", "string"], ["in a", "list"]]
    })
    actual = example.apply(parse_amenity_field, args=['amenities'], axis=1)
    expected = pd.Series([["some random", "string"], ["in a", "list"]])
    assert expected.equals(actual)


def test_parse_amenity_field_str():
    example = pd.DataFrame({
        'amenities': ['["some random", "string"]', '["in a", "list"]']
    })
    actual = example.apply(parse_amenity_field, args=['amenities'], axis=1)
    expected = pd.Series([["some random", "string"], ["in a", "list"]])
    assert expected.equals(actual)


def test_has_amenities_true():
    actual = AMENITIES_EXAMPLE.apply(has_amenity, args=['amenities', 'washer'], axis=1)
    expected = pd.Series([True])
    assert actual.equals(expected)


def test_has_amenities_false():
    actual = AMENITIES_EXAMPLE.apply(has_amenity, args=['amenities', 'random amenity'], axis=1)
    expected = pd.Series([False])
    assert actual.equals(expected)


def test_has_air_conditioning():
    actual = AMENITIES_EXAMPLE.apply(has_air_conditioning, args=['amenities'], axis=1)
    expected = pd.Series([True])
    assert actual.equals(expected)


def test_has_breakfast():
    actual = AMENITIES_EXAMPLE.apply(has_breakfast, args=['amenities'], axis=1)
    expected = pd.Series([False])
    assert actual.equals(expected)


def test_has_kitchen():
    actual = AMENITIES_EXAMPLE.apply(has_kitchen, args=['amenities'], axis=1)
    expected = pd.Series([True])
    assert actual.equals(expected)


def test_has_heating():
    actual = AMENITIES_EXAMPLE.apply(has_heating, args=['amenities'], axis=1)
    expected = pd.Series([True])
    assert actual.equals(expected)


def test_has_internet():
    actual = AMENITIES_EXAMPLE.apply(has_internet, args=['amenities'], axis=1)
    expected = pd.Series([False])
    assert actual.equals(expected)


def test_has_tv():
    actual = AMENITIES_EXAMPLE.apply(has_tv, args=['amenities'], axis=1)
    expected = pd.Series([True])
    assert actual.equals(expected)


def test_has_wifi():
    actual = AMENITIES_EXAMPLE.apply(has_wifi, args=['amenities'], axis=1)
    expected = pd.Series([True])
    assert actual.equals(expected)


def test_get_amenties_available():
    actual = get_amenties_available(df=AMENITIES_EXAMPLE, amenities_feature='amenities')
    expected = pd.DataFrame({
        'air_conditioning': [True],
        'breakfast': [False],
        'elevator': [True],
        'heating': [True],
        'internet': [False],
        'kitchen': [True],
        'tv': [True],
        'wifi': [True]
    })
    assert expected.equals(actual)


def test_remove_records_below_threshold_price():
    actual = remove_records_below_threshold_price(df=PRICING_EXAMPLE, price_feature='price', price_threshold=98)
    expected = pd.DataFrame({'price': [99]}, index=[4])
    assert expected.equals(actual)


def test_create_categorical_price_labels():
    actual = create_categorical_price_labels(
        df=PRICING_EXAMPLE,
        price_feature='price',
        bins=[0, 10, 20, 30, 100],
        labels=[0, 1, 2, 3]
    )
    expected = pd.DataFrame({
        'price': PRICING_EXAMPLE['price'].to_list(),
        'category': pd.Categorical([1, 0, 0, 2, 3, 3, 2, 3], [0, 1, 2, 3], ordered=True)
    })
    assert expected.equals(actual)


def test_map_categorical_features():
    mapping = {
        'vehicle': {'car': 1, 'bike': 2, 'boat': 0},
        'animal': {'cat': 2, 'dog': 0, 'frog': 1}
    }
    example = pd.DataFrame({
        'vehicle': ['car', 'car', 'boat', 'bike'],
        'animal': ['cat', 'dog', 'dog', 'frog']
    })
    actual = map_categorical_features(features_to_map=mapping, df=example)
    expected = pd.DataFrame({
        'vehicle': [1, 1, 0, 2],
        'animal': [2, 0, 0, 1]
    })
    assert expected.equals(actual)
