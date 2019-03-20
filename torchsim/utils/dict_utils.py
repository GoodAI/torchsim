import dataclasses
from typing import Any


class NestedDictException(Exception):
    pass


def get_dict_intersection(dict1: dict, dict2: dict):
    """Gets the common items of two nested dicts."""
    result = {}
    for key, value1 in dict1.items():
        if key not in dict2:
            continue

        value2 = dict2[key]

        # Add support for list/tuple here if needed.
        if type(value1) is dict and type(value2) is dict:
            inner_result = get_dict_intersection(value1, value2)
            if inner_result:
                result[key] = inner_result
        elif value1 == value2:
            result[key] = value1

    return result


def dict_with_defaults(data: dict, defaults: dict):
    """Creates a dict which has values from defaults updated by values in data.

    The two dicts must have the same nested structure."""

    result = dict(defaults)
    for data_key, data_value in data.items():
        if data_key not in defaults:
            raise NestedDictException(f"Key {data_key} not present in defaults")

        default_value = defaults[data_key]
        if type(default_value) is dict:
            if type(data_value) is not dict:
                raise NestedDictException(f"Item with key {data_key} is a dict in defaults but not in data")

            result[data_key] = dict_with_defaults(data_value, default_value)
        else:
            result[data_key] = data_value

    return result


def remove_from_dict(dict1: dict, dict2: dict):
    """Removes the values from a nested dict1 which match the values in a nested dict2."""
    processed_keys = set()
    result = {}
    for key2, value2 in dict2.items():
        value1 = dict1.get(key2, None)
        if value1 is None:
            # The value is already missing.
            continue

        if value1 == value2:
            # The value needs to be missing.
            processed_keys.add(key2)
            continue

        if type(value1) == dict and type(value2) == dict:
            # The dicts are different because of the equality check above.
            # Process them recursively.
            result[key2] = remove_from_dict(value1, value2)
            processed_keys.add(key2)

    # Add items which were not in dict2.
    for key1, value1 in dict1.items():
        if key1 not in processed_keys:
            result[key1] = value1

    return result


def to_nested_dict(data: Any) -> dict:
    """Converts dataclasses and namedtuples in data to dictionaries.

    Args:
        data: A dict, a namedtuple, or a dataclass. Both can be nested.
    """
    if dataclasses.is_dataclass(data):
        return to_nested_dict(dataclasses.asdict(data))
    elif hasattr(data, '_asdict'):
        return to_nested_dict(data._asdict())
    elif isinstance(data, dict):
        return {key: to_nested_dict(value) for key, value in data.items()}
    else:
        return data
