import operator
from functools import reduce

from abc import abstractmethod, ABC
from typing import List, Tuple

from torchsim.core.exceptions import IllegalArgumentException
from torchsim.core.graph.node_base import NodeBase
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit


def recursive_get_nodes(node: NodeBase) -> List[NodeBase]:
    if isinstance(node, NodeGroupBase):
        return [node, *(i for n in node.nodes for i in recursive_get_nodes(n))]
    else:
        return [node]


def derive_flock_shape(whole_input_shape: Tuple[int, ...], flock_size: int) -> Tuple[int, ...]:
    """Automatically derive the flock shape dimensions.

    Automatically derive the flock shape (dimension of the mesh on which experts are located) from the first m
    dims of flock and reshape the outputs accordingly.
    Example: input.shape == (10,10,5,2,2,3), flock_size == 500
                 from that 500 == 10*10*5 is flock shape and so 2*2*3 is input of single expert.
             output.shape == (flock_shape, n_cluster_centers) == (10, 10, 5, n_cluster_centers)

    Without this code output shape would be flat:
        output.shape == (500,n_cluster_centers)

    Note: In case it is not clear if a singleton dimension belongs to the flock_shape or input_to_one_expert_shape,
    it belongs to the flock_shape only if the whole_input_shape starts with 1. So, for example:
        input.shape == (4, 4, 1, 5), flock_size == 16 -> flock_shape = (4, 4), input_to_one_expert_shape = (1, 5)
        input.shape == (4, 1, 4, 5), flock_size == 16 -> flock_shape = (4, 1, 4), input_to_one_expert_shape = (5,)
        input.shape == (1, 1, 5), flock_size == 1 -> flock_shape = (1), input_to_one_expert_shape = (1, 5)

    Args:
        whole_input_shape: Shape of the data_input to the expert.
        flock_size: Number of experts in the flock.

    Returns:
        Flock shape - shape of the mesh on which the flocks are located.
    """

    def raise_exception():
        raise IllegalArgumentException(
            f"Don't know how to distribute the data {whole_input_shape} among {flock_size} experts, "
            f"the product of first n dimensions of the data should be equal to the flock_size.")

    Sum = TensorShapePatternMatcher.Sum
    dim_product = reduce(operator.mul, whole_input_shape, 1)
    if dim_product % flock_size != 0:
        raise_exception()
    matcher = TensorShapePatternMatcher((Sum(flock_size), Sum(dim_product // flock_size, greedy=True)))
    if matcher.matches(whole_input_shape):
        return matcher.groups[0]
    else:
        raise_exception()


class TensorShapePatternMatcher:
    """Simplified regular-like expressions for tensor shapes.

    Only Sum and Exact matchers are supported.

    TODO: This class can be extended to behave like regular expressions and support matcher Any (arbitrary number of any values)
    """

    class PatternItem(ABC):
        @abstractmethod
        def consume(self, value: int) -> bool:
            pass

        @abstractmethod
        def is_accepted(self) -> bool:
            pass

        @abstractmethod
        def reset(self):
            pass

    class Sum(PatternItem):
        """Cumulative product matcher - accepts values until cumulative product of dimensions equals expected_value
        Trailing ones are accepted only when greedy is True

        Examples:
            Sum(12) accepts following shapes: (2,6), (12,), (1,1,12), (2,3,2)
            Sum(12, greedy=True) accepts following shapes: (2,6,1), (12,1,1,1)
        """
        _cumulative_sum: int
        _any_value_accepted: bool

        def __init__(self, expected_value: int, greedy: bool = False):
            self._expected_sum = expected_value
            self._greedy = greedy

        def reset(self):
            self._cumulative_sum = 1
            self._any_value_accepted = False

        def consume(self, value: int) -> bool:
            if not self._greedy and value == 1 and self.is_accepted():
                # Do not accept trailing ones
                return False

            cumulative_sum = self._cumulative_sum * value
            if cumulative_sum <= self._expected_sum:
                self._cumulative_sum = cumulative_sum
                self._any_value_accepted = True
                return True
            else:
                return False

        def is_accepted(self) -> bool:
            return self._any_value_accepted and self._expected_sum == self._cumulative_sum

        def __str__(self) -> str:
            return f'Sum({self._expected_sum}{", greedy=True" if self._greedy else ""})'

    class Exact(PatternItem):
        """Exact shape matcher - matches only the exact_shape

        Examples:
            Exact((1,4,5)) accepts just shape: (1,4,5)
        """
        pos: int
        mismatch: bool

        def __init__(self, expected_shape: Tuple[int, ...]):
            self._expected_shape = expected_shape

        def reset(self):
            self.mismatch = False
            self.pos = 0

        def consume(self, value: int) -> bool:
            if self.pos >= len(self._expected_shape):
                return False
            elif value == self._expected_shape[self.pos]:
                self.pos += 1
                return True
            else:
                self.mismatch = True
                return False

        def is_accepted(self) -> bool:
            return not self.mismatch and self.pos == len(self._expected_shape)

        def __str__(self) -> str:
            params = ", ".join(map(str, self._expected_shape))
            return f'Exact({params})'

    class TrailingAny(PatternItem):

        def consume(self, value: int) -> bool:
            return True

        def is_accepted(self) -> bool:
            return True

        def reset(self):
            pass

        def __str__(self) -> str:
            return f'TrailingAny()'

    _groups: List[Tuple[int, ...]]

    def __init__(self, pattern: Tuple[PatternItem, ...]):
        self._pattern = pattern
        self._groups = []

    def matches(self, shape: Tuple[int, ...]) -> bool:
        """Check if passed shape matches the pattern"""
        pattern_pos = 0
        self._groups = []
        current_group: List[int] = []
        for item in self._pattern:
            item.reset()

        for dim in shape:
            while True:
                consumed = self._pattern[pattern_pos].consume(dim)
                if not consumed:
                    self._groups.append(tuple(current_group))
                    current_group = []
                    pattern_pos += 1
                    if pattern_pos >= len(self._pattern):
                        return False
                else:
                    current_group.append(dim)
                    break
        self._groups.append(tuple(current_group))
        all_accepted = all([i.is_accepted() for i in self._pattern])
        return all_accepted and pattern_pos == len(self._pattern) - 1

    @property
    def groups(self) -> List[Tuple[int, ...]]:
        """Get list of dimensions accepted by particular matchers
        This method must be used after matches()

         Examples:
             matcher = TensorShapePatternMatcher(Sum(6), Sum(10))
             matcher.matches((2,3,2,5))
             assert [(2,3), (2,5)] == matcher.groups
         """
        return self._groups

    @property
    def pattern(self) -> Tuple[PatternItem, ...]:
        return self._pattern


class TestMemoryBlocks(MemoryBlocks):
    def prepare_slots(self, unit: Unit):
        pass
