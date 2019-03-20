from dataclasses import dataclass
from typing import Union, Any, Optional

import pytest

from torchsim.utils.dict_utils import get_dict_intersection, NestedDictException, dict_with_defaults, remove_from_dict, \
    to_nested_dict


@dataclass
class TwoDictsTestItem:
    dict1: dict
    dict2: dict
    expected: Union[dict, pytest.raises]


@pytest.mark.parametrize('test_item', [
    TwoDictsTestItem(dict1={'a': 1}, dict2={'a': 1}, expected={'a': 1}),
    TwoDictsTestItem(dict1={'a': 1}, dict2={'b': 1}, expected={}),
    TwoDictsTestItem(dict1={'a': 1}, dict2={'b': 2}, expected={}),
    TwoDictsTestItem(dict1={'a': 1, 'b': 2}, dict2={'a': 1, 'b': 3}, expected={'a': 1}),
    TwoDictsTestItem(dict1={'a': 1, 'b': {'c': 2}},
                     dict2={'a': 1, 'b': {'d': 2}},
                     expected={'a': 1}),
    TwoDictsTestItem(dict1={'a': 1, 'b': {'c': 2}},
                     dict2={'a': 1, 'b': {'c': 3}},
                     expected={'a': 1}),
    TwoDictsTestItem(dict1={'a': 1, 'b': {'c': 2}},
                     dict2={'a': 1, 'b': {'c': 2}},
                     expected={'a': 1, 'b': {'c': 2}}),
    TwoDictsTestItem(dict1={'a': 1, 'b': {'c': 2}, 'd': 3},
                     dict2={'a': 1, 'b': {'c': 2}, 'd': 4},
                     expected={'a': 1, 'b': {'c': 2}}),
    TwoDictsTestItem(dict1={'a': 1, 'b': {'c': 2, 'd': 3}},
                     dict2={'a': 1, 'b': {'c': 2, 'd': 4}},
                     expected={'a': 1, 'b': {'c': 2}})
])
def test_dict_intersection(test_item):
    assert test_item.expected == get_dict_intersection(test_item.dict1, test_item.dict2)


@pytest.mark.parametrize('test_item', [
    TwoDictsTestItem(dict1={'a': 1}, dict2={'a': 1}, expected={'a': 1}),
    TwoDictsTestItem(dict1={}, dict2={'a': 1}, expected={'a': 1}),
    TwoDictsTestItem(dict1={'a': 1}, dict2={'a': 2}, expected={'a': 1}),
    TwoDictsTestItem(dict1={'a': {'b': 1}}, dict2={'a': {'b': 1}}, expected={'a': {'b': 1}}),
    TwoDictsTestItem(dict1={'a': {'b': 1}}, dict2={'a': {'b': 2}}, expected={'a': {'b': 1}}),
    TwoDictsTestItem(dict1={}, dict2={'a': {'b': 1}}, expected={'a': {'b': 1}}),
    TwoDictsTestItem(dict1={'b': 3}, dict2={'a': {'b': 1}, 'b': 2}, expected={'a': {'b': 1}, 'b': 3}),
    TwoDictsTestItem(dict1={'a': 3}, dict2={'a': {'b': 1}}, expected=pytest.raises),
    TwoDictsTestItem(dict1={'b': 3}, dict2={'a': 1}, expected=pytest.raises),
])
def test_dict_with_defaults(test_item):
    if test_item.expected == pytest.raises:
        with pytest.raises(NestedDictException):
            dict_with_defaults(test_item.dict1, test_item.dict2)
    else:
        assert test_item.expected == dict_with_defaults(test_item.dict1, test_item.dict2)


@pytest.mark.parametrize('test_item', [
    TwoDictsTestItem(dict1={'a': 1}, dict2={}, expected={'a': 1}),
    TwoDictsTestItem(dict1={'a': 1}, dict2={'a': 2}, expected={'a': 1}),
    TwoDictsTestItem(dict1={'a': 1}, dict2={'b': 1}, expected={'a': 1}),
    TwoDictsTestItem(dict1={'a': 1}, dict2={'a': 1}, expected={}),
    TwoDictsTestItem(dict1={'a': {'b': 2}}, dict2={'a': {'b': 2}}, expected={}),
    TwoDictsTestItem(dict1={'a': {'b': 2}}, dict2={'a': {'c': 2}}, expected={'a': {'b': 2}}),
    TwoDictsTestItem(dict1={'a': {'b': 2}}, dict2={'a': {'b': 1}}, expected={'a': {'b': 2}}),
    TwoDictsTestItem(dict1={'a': 3}, dict2={'a': {'b': 1}}, expected={'a': 3}),
    TwoDictsTestItem(dict1={'a': {'b': 1}}, dict2={'a': 3}, expected={'a': {'b': 1}}),
])
def test_remove_from_dict(test_item):
    assert test_item.expected == remove_from_dict(test_item.dict1, test_item.dict2)


@dataclass
class DataclassStub:
    inner: Optional['DataclassStub'] = None


@dataclass
class DataClassTestItem:
    data: Union[dict, DataclassStub]
    expected: dict


@pytest.mark.parametrize('test_item', [
    DataClassTestItem(data={}, expected={}),
    DataClassTestItem(data=DataclassStub(), expected={'inner': None}),
    DataClassTestItem(data={'data': DataclassStub()}, expected={'data': {'inner': None}}),
    DataClassTestItem(data={'data': DataclassStub(), 'data2': None},
                      expected={'data': {'inner': None}, 'data2': None}),
    DataClassTestItem(data={'data': DataclassStub(inner=DataclassStub())},
                      expected={'data': {'inner': {'inner': None}}}),
])
def test_dataclasses_to_dict(test_item):
    assert test_item.expected == to_nested_dict(test_item.data)
