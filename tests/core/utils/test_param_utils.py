from typing import NamedTuple, List

from torchsim.utils.dict_utils import to_nested_dict


class NestedParams(NamedTuple):
    width: int
    items: List[int]


class Params(NamedTuple):
    name: str
    projection: NestedParams


def test_namedtuple_to_dict_simple():
    nested = NestedParams(10, [1, 2, 3])
    result = to_nested_dict(nested)
    assert {
               'width': 10,
               'items': [1, 2, 3]
           } == result


def test_namedtuple_to_dict_nested():
    nested = NestedParams(10, [1, 2, 3])
    params = Params('first', nested)

    result = to_nested_dict(params)
    assert {
               'name': 'first',
               'projection': {
                   'width': 10,
                   'items': [1, 2, 3]
               }
           } == result
