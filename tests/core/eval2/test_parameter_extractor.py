from dataclasses import dataclass
from typing import List, Dict, Any

import pytest

from torchsim.core.eval2.parameter_extractor import ParameterExtractor


@dataclass
class TestItem:
    parameters: List[Dict[str, Any]]
    default_parameters: Dict[str, Any]
    expected_header: Dict[str, Any]
    expected_legend: List[Dict[str, Any]]


@pytest.mark.parametrize('test_item', [
    # All parameters are the same.
    TestItem(parameters=[{'a': 1,
                          'b': 2},
                         {'a': 1,
                          'b': 2}],
             default_parameters={'a': 1,
                                 'b': 2},
             expected_header={'a': 1,
                              'b': 2},
             expected_legend=[{}, {}]),
    # All defaults.
    TestItem(parameters=[{}, {}],
             default_parameters={'a': 1,
                                 'b': {'c': 2}},
             expected_header={'a': 1,
                              'b': {'c': 2}},
             expected_legend=[{}, {}]),
    # One item is the same but different from default, the other differs.
    TestItem(parameters=[{'a': 1,
                          'b': 2},
                         {'a': 1,
                          'b': 3}],
             default_parameters={'a': 3,
                                 'b': 2},
             expected_header={'a': 1},
             expected_legend=[{'b': 2},
                              {'b': 3}]),
    # All items are different from default but they match.
    TestItem(parameters=[{'a': 2,
                          'b': 3},
                         {'a': 2,
                          'b': 3}],
             default_parameters={'a': 1,
                                 'b': 2},
             expected_header={'a': 2, 'b': 3},
             expected_legend=[{}, {}]),
    # Values are missing in some runs, defaults should be used.
    TestItem(parameters=[{'a': 2,
                          'b': 3},
                         {'b': 3}],
             default_parameters={'a': 1,
                                 'b': 2},
             expected_header={'b': 3},
             expected_legend=[{'a': 2}, {'a': 1}]),
    # A nested parameter is shared but others are not.
    TestItem(parameters=[{'a': 1,
                          'b': {'c': 2}},
                         {'a': 2,
                          'b': {'d': 5}}],
             default_parameters={'a': 2,
                                 'b': {'c': 3, 'd': 5}},
             expected_header={'b': {'d': 5}},
             expected_legend=[{'a': 1, 'b': {'c': 2}},
                              {'a': 2, 'b': {'c': 3}}]),
])
def test_parameters(test_item):
    parameter_extractor = ParameterExtractor(test_item.default_parameters)

    header, legend = parameter_extractor.extract(test_item.parameters)

    assert test_item.expected_header == header
    assert test_item.expected_legend == legend
