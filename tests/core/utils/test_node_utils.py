import pytest
from pytest import raises
from typing import List

from torchsim.core.exceptions import IllegalArgumentException
from torchsim.core.graph.node_base import NodeBase
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes import JoinNode
from torchsim.utils.node_utils import recursive_get_nodes, derive_flock_shape, TensorShapePatternMatcher


class Group(NodeGroupBase):
    def __init__(self, name: str, nodes: List[NodeBase]):
        super().__init__(name)
        self.nodes = nodes


class TestRecursiveGetNodes:
    def test_recursive_get_nodes(self):
        # simple node
        node = JoinNode()
        assert [node] == recursive_get_nodes(node)

    def test_single_layer(self):
        n1 = JoinNode()
        n2 = JoinNode()
        n3 = JoinNode()
        g1 = Group("g", [n1, n2, n3])
        assert [g1, n1, n2, n3] == recursive_get_nodes(g1)

    def test_two_layer(self):
        n1 = JoinNode()
        n2 = JoinNode()
        n3 = JoinNode()
        n4 = JoinNode()
        g1 = Group("g1", [n2, n3])
        g2 = Group("g2", [n1, g1, n4])
        assert [g1, n2, n3] == recursive_get_nodes(g1)
        assert [g2, n1, g1, n2, n3, n4] == recursive_get_nodes(g2)


class TestDeriveFlockShape:
    @pytest.mark.parametrize('input_shape, flock_size, expected_flock_shape', [
        ((2, 3, 4, 5), 2, (2,)),
        ((2, 3, 4, 5), 6, (2, 3)),
        ((2, 3, 4, 5), 24, (2, 3, 4)),
        ((1, 2, 1, 2, 1), 1, (1,)),
        ((1, 2, 1, 2, 1), 2, (1, 2)),
        ((1, 2, 1, 2, 1), 4, (1, 2, 1, 2)),
    ])
    def test_derive_flock_shape(self, input_shape, flock_size, expected_flock_shape):
        result = derive_flock_shape(input_shape, flock_size)
        assert expected_flock_shape == result

    @pytest.mark.parametrize('input_shape, flock_size', [
        ((2, 3, 4, 5), 1,),
        ((2, 3, 4, 5), 3,),
        ((2, 3, 4, 5), 4,),
        ((2, 3, 4, 5), 5,),
        ((2, 3, 4, 5), 7,)
    ])
    def test_derive_flock_shape_exception(self, input_shape, flock_size):
        with raises(IllegalArgumentException):
            derive_flock_shape(input_shape, flock_size)


Sum = TensorShapePatternMatcher.Sum
Exact = TensorShapePatternMatcher.Exact
TrailingAny = TensorShapePatternMatcher.TrailingAny


class TestTensorShapePatternMatcher:
    @pytest.mark.parametrize('shape, pattern, is_valid', [
        ((2, 3, 4), (Sum(24),), True),
        ((2, 3, 4), (Sum(6), Exact((4,))), True),
        ((2, 3, 4), (Sum(2), Exact((3, 4))), True),
        ((2, 3, 4), (Exact((2, 3, 4)),), True),
        ((2, 3, 4), (Sum(23),), False),
        ((2, 3, 4), (Sum(25),), False),
        ((2, 3, 4), (Exact((2, 3)),), False),
        ((2, 3, 4), (Exact((2, 3, 5)),), False),
        ((2, 3, 4), (Exact((2, 3, 4, 5)),), False),
        ((2, 3, 4), (Sum(5), Exact((4,))), False),
        ((2, 3, 4), (Sum(6), Exact((3,))), False),
        ((2, 3, 4, 2, 2, 4), (Sum(6), Exact((4, 2)), Sum(8)), True),
        ((2, 3, 4), (Sum(24), Exact((1,))), False),
        ((1, 2, 3, 4), (Sum(6), Exact((4,))), True),
        ((1, 2, 3, 1, 4), (Sum(6), Exact((1, 4))), True),
        ((1, 2, 3, 1, 1, 4), (Sum(6), Exact((1, 1, 4))), True),  # ones are not consumed
        ((2, 3), (Sum(1), Exact((2, 3))), False),
        ((1, 3), (Sum(1), Exact((3,))), True),
        ((2, 3, 4), (Sum(24), TrailingAny()), False),
        ((2, 3, 4, 1), (Sum(24), TrailingAny()), True),
        ((2, 3, 4, 1, 2), (Sum(24), TrailingAny()), True),
    ])
    def test_match(self, shape, pattern, is_valid):
        matcher = TensorShapePatternMatcher(pattern)
        assert is_valid == matcher.matches(shape)

    @pytest.mark.parametrize('shape, pattern, is_valid', [
        ((1, 2, 3, 1, 1, 4), (Sum(6), Exact((1, 1, 4))), True),  # ones are not consumed
        ((2, 3), (Sum(1), Exact((2, 3))), False),
        ((1, 3), (Sum(1), Exact((3,))), True),
        ((1, 1, 1, 3), (Sum(1, greedy=True), Exact((3,))), True),
        ((4, 1, 1, 3), (Sum(4, greedy=True), Exact((3,))), True),
        ((4, 1, 1, 3, 1, 2, 1, 1), (Sum(4, greedy=True), Exact((3,)), Sum(2, greedy=True)), True),
    ])
    def test_match_sum(self, shape, pattern, is_valid):
        matcher = TensorShapePatternMatcher(pattern)
        assert is_valid == matcher.matches(shape)

    @pytest.mark.parametrize('shape, pattern, groups', [
        ((1, 2, 3, 4, 5), (Sum(2), Sum(60)), [(1, 2), (3, 4, 5)]),
        ((1, 2, 3, 4, 5), (Sum(6), Sum(20)), [(1, 2, 3), (4, 5)]),
        ((1, 2, 3, 4, 5), (Sum(24), Sum(5)), [(1, 2, 3, 4), (5,)]),
        ((1, 2, 3, 4, 5), (Exact((1, 2)), Sum(12), Exact((5,))), [(1, 2), (3, 4), (5,)]),
        ((1, 2, 3, 4, 5), (Sum(2), TrailingAny()), [(1, 2), (3, 4, 5)]),
    ])
    def test_groups(self, shape, pattern, groups):
        matcher = TensorShapePatternMatcher(pattern)
        assert True is matcher.matches(shape)
        assert groups == matcher.groups

    @pytest.mark.parametrize('pattern_item, expected_str', [
        (Sum(1), 'Sum(1)'),
        (Sum(15, greedy=True), 'Sum(15, greedy=True)'),
        (Sum(23, greedy=False), 'Sum(23)'),
        (Exact((3,)), 'Exact(3)'),
        (Exact((3, 5, 18)), 'Exact(3, 5, 18)'),
        (TrailingAny(), 'TrailingAny()'),
    ])
    def test_pattern_items_str(self, pattern_item, expected_str):
        assert expected_str == str(pattern_item)
