from abc import abstractmethod, ABC
from enum import Enum
from pytest import raises
from typing import Tuple, Generator, List, Any

import pytest
import torch

from torchsim.core import get_float
from torchsim.core.graph import Topology
from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket
from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.nodes import ConvExpertFlockNode
from torchsim.core.nodes.expert_node import ExpertFlockNode
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.memory.tensor_creator import MeasuringCreator, TensorSurrogate
from torchsim.core.models.expert_params import ExpertParams, NUMBER_OF_CONTEXT_TYPES
from torchsim.core.utils.tensor_utils import same
from tests.core.nodes.node_unit_test_base import NodeTestBase


# region test inputs combinations


class RewardInputType(Enum):
    SCALAR_POS = 0,
    FULL = 1,
    NONE = 2,
    PAIR = 3,
    SCALAR_NEG = 4



class InputCombinationsBase(NodeTestBase, ABC):
    """Tests different combinations of inputs to ExpertFlockNode."""

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class()
        cls._dim = 1
        cls._device = 'cuda'
        cls.params = ExpertParams()
        cls.params.n_cluster_centers = 4
        cls.params.flock_size = 3
        cls.params.spatial.input_size = 5
        cls.params.spatial.buffer_size = 10
        cls.params.spatial.batch_size = 5
        cls.params.temporal.buffer_size = 11
        cls.params.temporal.batch_size = 5
        cls.params.temporal.n_frequent_seqs = 3
        cls.params.temporal.max_encountered_seqs = 7
        cls.params.temporal.incoming_context_size = cls._get_context_size()

    @staticmethod
    def skip_test_serialization():
        """It's enough to test it just in with full context and rewards."""
        return True

    @classmethod
    @abstractmethod
    def use_inputs(cls) -> Tuple[bool, RewardInputType]:
        """Connect context input?, Connect reward input?"""
        pass

    @classmethod
    def _get_context_size(cls) -> int:
        use_context, use_rewards = cls.use_inputs()
        if use_context:
            return 4
        else:
            return 1

    def _generate_input_tensors(self):

        input_tensor = self._creator.zeros((self.params.flock_size, self.params.spatial.input_size),
                                           device=self._device,
                                           dtype=self._dtype)

        use_context, use_rewards = self.use_inputs()
        if use_context:
            context_tensor = self._creator.full((self.params.flock_size, self.params.temporal.n_providers, NUMBER_OF_CONTEXT_TYPES,
                                                 self.params.temporal.incoming_context_size), fill_value=0.7,
                                                device=self._device,
                                                dtype=self._dtype)
            context_tensor[:, :, 1:, :] = 0

        else:
            context_tensor = None

        if use_rewards == RewardInputType.SCALAR_POS:
            reward_tensor = self._creator.full((1,), fill_value=0.61, device=self._device, dtype=self._dtype)
        elif use_rewards == RewardInputType.SCALAR_NEG:
            reward_tensor = self._creator.full((1,), fill_value=-0.32, device=self._device, dtype=self._dtype)
        elif use_rewards == RewardInputType.FULL:
            reward_tensor = self._creator.full((self.params.flock_size, 2), fill_value=0.3, device=self._device,
                                               dtype=self._dtype)
            reward_tensor[:, 1] = 0.7
        elif use_rewards == RewardInputType.PAIR:
            reward_tensor = self._creator.tensor([0.6, 0.1], device=self._device, dtype=self._dtype)
        else:
            reward_tensor = None

        yield [input_tensor, context_tensor, reward_tensor]

    def _generate_expected_results(self):
        # Testing just that it combines the rewards and contexts correctly

        dtype = get_float(self._device)
        flock_size = self.params.flock_size

        self.context_tensor = torch.zeros((flock_size, self.params.temporal.n_providers, NUMBER_OF_CONTEXT_TYPES, self.params.temporal.incoming_context_size),
                                          dtype=dtype, device=self._device)

        # default context vector if neither context nor rewards are connected
        self.context_tensor[:, :, 0, :] = 1 / self.params.n_cluster_centers
        self.context_tensor[:, :, 1, :] = 0
        self.context_tensor[:, :, 2, :] = 0

        use_context, use_rewards = self.use_inputs()
        if use_context:
            self.context_tensor[:, :, 0, :] = 0.7

        if use_rewards == RewardInputType.FULL:
            self.reward_tensor = torch.zeros((flock_size, 2), device=self._device, dtype=self._dtype)
            self.reward_tensor[:, 0] = 0.3
            self.reward_tensor[:, 1] = 0.7
        elif use_rewards == RewardInputType.SCALAR_POS:
            self.reward_tensor = torch.zeros((flock_size, 2), device=self._device, dtype=self._dtype)
            self.reward_tensor[:, 0] = 0.61
        elif use_rewards == RewardInputType.SCALAR_NEG:
            self.reward_tensor = torch.zeros((flock_size, 2), device=self._device, dtype=self._dtype)
            self.reward_tensor[:, 1] = 0.32
        elif use_rewards == RewardInputType.PAIR:
            self.reward_tensor = torch.zeros((flock_size, 2), device=self._device, dtype=self._dtype)
            self.reward_tensor[:, 0] = 0.6
            self.reward_tensor[:, 1] = 0.1
        else:
            self.reward_tensor = torch.zeros((flock_size, 2), device=self._device, dtype=self._dtype)

        yield [self.context_tensor, self.reward_tensor]

    def _create_node(self):
        return ExpertFlockNode(params=self.params.clone())

    def _extract_results(self, node):
        return [node._unit.flock.tp_flock.input_context, node._unit.flock.tp_flock.input_rewards]


class TestInputContextFull(InputCombinationsBase):
    @staticmethod
    def skip_test_serialization():
        """Test serialization just on this combination."""
        return False

    @classmethod
    def use_inputs(cls):
        return True, RewardInputType.FULL


class TestInputFull(InputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return False, RewardInputType.FULL


class TestInputContextScalarPos(InputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return True, RewardInputType.SCALAR_POS


class TestInputScalarPos(InputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return False, RewardInputType.SCALAR_POS


class TestInputContextScalarNeg(InputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return True, RewardInputType.SCALAR_NEG


class TestInputScalarNeg(InputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return False, RewardInputType.SCALAR_NEG


class TestInputContextPair(InputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return True, RewardInputType.PAIR


class TestInputPair(InputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return False, RewardInputType.PAIR


class TestInputContextNone(InputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return True, RewardInputType.NONE


class TestInputNone(InputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return False, RewardInputType.NONE


# region ConvExpertFlockNode

class ConvInputCombinationsBase(InputCombinationsBase):
    """Teting also Conv version."""

    @classmethod
    @abstractmethod
    def use_inputs(cls) -> Tuple[bool, RewardInputType]:
        """Connect context input?, Connect reward input?"""
        pass

    def _create_node(self):
        return ConvExpertFlockNode(params=self.params.clone())


class TestConvInputContextFull(ConvInputCombinationsBase):
    @staticmethod
    def skip_test_serialization():
        """Test serialization just on this combination."""
        return False

    @classmethod
    def use_inputs(cls):
        return True, RewardInputType.FULL


class TestConvInputFull(ConvInputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return False, RewardInputType.FULL


class TestConvInputContextNone(ConvInputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return True, RewardInputType.NONE


class TestConvInputNone(ConvInputCombinationsBase):
    @classmethod
    def use_inputs(cls):
        return False, RewardInputType.NONE


# endregion

# endregion


class SerializationTestBase(NodeTestBase, ABC):
    """Tests just the serialization of the 'enable learning' flags for now."""

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class()
        cls._dim = 1
        cls._device = 'cuda'
        cls.params = ExpertParams()
        cls.params.flock_size = 3
        cls.params.spatial.input_size = 5

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        input_tensor = self._creator.zeros((self.params.flock_size, self.params.spatial.input_size),
                                           device=self._device,
                                           dtype=self._dtype)
        yield [input_tensor, None, None]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        yield [self._get_learning_combination()] * 2

    def _create_node(self):
        params = self.params.clone()
        params.spatial.enable_learning, params.temporal.enable_learning = self._get_learning_combination()

        return ExpertFlockNode(params)

    @abstractmethod
    def _get_learning_combination(self):
        pass

    def _extract_results(self, node):
        return [(node._unit.flock.sp_flock.enable_learning, node._unit.flock.tp_flock.enable_learning),
                (node.params.spatial.enable_learning, node.params.temporal.enable_learning)]

    def _change_node_before_load(self, node):
        node.params.spatial.enable_learning, node.params.temporal.enable_learning = False, False
        node._unit.flock.sp_flock.enable_learning, node._unit.flock.tp_flock.enable_learning = False, False


class TestLearningSerializationSP(SerializationTestBase):
    def _get_learning_combination(self):
        return True, False


class TestLearningSerializationTP(SerializationTestBase):
    def _get_learning_combination(self):
        return False, True


class TestLearningSerializationBoth(SerializationTestBase):
    def _get_learning_combination(self):
        return True, True


# region ConvExpertFlockNode

class ConvSerializationTestBase(SerializationTestBase):
    """Tests Conv version of the expert"""

    def _create_node(self):
        params = self.params.clone()
        params.spatial.enable_learning, params.temporal.enable_learning = self._get_learning_combination()

        return ConvExpertFlockNode(params)

    @abstractmethod
    def _get_learning_combination(self):
        pass


class TestConvLearningSerializationSP(ConvSerializationTestBase):
    def _get_learning_combination(self):
        return True, False


class TestConvLearningSerializationTP(ConvSerializationTestBase):
    def _get_learning_combination(self):
        return False, True


class TestConvLearningSerializationBoth(ConvSerializationTestBase):
    def _get_learning_combination(self):
        return True, True


# endregion


class TestExpertFlockNode:
    @staticmethod
    def memory_block(tensor: torch.Tensor) -> MemoryBlock:
        memory_block = MemoryBlock()
        memory_block.tensor = tensor
        return memory_block

    @pytest.mark.parametrize('sp_data_input_shape, flock_size, context_size, context_input_shape, is_valid', [
        ((3, 2, 100), 6, 10, (6, 1, 3, 10), True),
        ((3, 2, 100), 6, 24, (6, 1, 3, 24), True),
        ((3, 2, 100), 6, 24, (6, 1, 3, 2, 12), True),
        ((3, 2, 100), 6, 24, (6, 1, 3, 2, 3, 4), True),
        ((3, 2, 100), 6, 24, (6, 1, 3, 2, 3, 2, 2), True),
        ((3, 2, 100), 6, 24, (3, 2, 3, 23), False),
        ((3, 2, 100), 6, 24, (3, 3, 3, 24), False),
    ])
    def test_validate_context_input(self, sp_data_input_shape, flock_size, context_size, context_input_shape, is_valid):
        params = ExpertParams()
        params.flock_size = flock_size
        params.temporal.incoming_context_size = context_size
        expert = ExpertFlockNode(params)
        Connector.connect(self.memory_block(torch.zeros(sp_data_input_shape)), expert.inputs.sp.data_input)
        Connector.connect(self.memory_block(torch.zeros(context_input_shape)), expert.inputs.tp.context_input)
        if is_valid:
            expert.validate()
        else:
            with raises(NodeValidationException):
                expert.validate()

    @pytest.mark.parametrize('sp_data_input_shape, flock_size, context_size, context_input_shape, exception_message', [
        ((3, 2, 100), 6, 24, (3, 2, 2, 23),
         r"Context input has unexpected shape \[3, 2, 2, 23\], expected pattern: \[Sum\(6\), Sum\(1, greedy=True\), Exact\(3\), Sum\(24, greedy=True\)\]"),
    ])
    def test_validate_context_input_error_message(self, sp_data_input_shape, flock_size, context_size,
                                                  context_input_shape, exception_message):
        params = ExpertParams()
        params.flock_size = flock_size
        params.temporal.incoming_context_size = context_size
        expert = ExpertFlockNode(params)
        Connector.connect(self.memory_block(torch.zeros(sp_data_input_shape)), expert.inputs.sp.data_input)
        Connector.connect(self.memory_block(torch.zeros(context_input_shape)), expert.inputs.tp.context_input)
        with raises(NodeValidationException, match=exception_message):
            expert.validate()

    def test_output_tensor_shapes(self):
        # Although the experts internally only work with all the input dimensions squashed into one, some memory blocks
        # should provide a view of the internal tensors in the same shape as the input that the node received.
        params = ExpertParams()
        input_size = (28, 28)

        node = ExpertFlockNode(params=params)

        input_block = MemoryBlock()
        input_block.tensor = TensorSurrogate(dims=(params.flock_size,) + input_size)
        Connector.connect(input_block, node.inputs.sp.data_input)

        tensor_creator = MeasuringCreator()
        node.allocate_memory_blocks(tensor_creator)

        assert input_size == tuple(node.memory_blocks.sp.buffer_inputs.shape[-2:])
        assert input_size == tuple(node.memory_blocks.sp.cluster_centers.shape[-2:])
        assert input_size == tuple(node.memory_blocks.sp.cluster_center_targets.shape[-2:])
        assert input_size == tuple(node.memory_blocks.sp.cluster_center_deltas.shape[-2:])

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_inverse_projection(self, device):
        dtype = get_float(device)
        params = ExpertParams()
        params.flock_size = 2
        params.n_cluster_centers = 4

        params.spatial.input_size = 6
        params.spatial.buffer_size = 7
        params.spatial.batch_size = 3
        params.temporal.n_frequent_seqs = 2
        params.temporal.seq_length = 3
        input_size = (3, 2)

        graph = Topology(device)
        node = ExpertFlockNode(params=params)

        graph.add_node(node)

        input_block = MemoryBlock()
        input_block.tensor = torch.rand((params.flock_size,) + input_size, dtype=dtype, device=device)
        Connector.connect(input_block, node.inputs.sp.data_input)

        graph.prepare()

        node._unit.flock.sp_flock.cluster_centers = torch.tensor([[[1, 0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 0],
                                                                   [0, 0, 0.5, 0.5, 0, 0],
                                                                   [0, 0, 0.5, 0, 0.5, 0]],
                                                                  [[1, 0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 0],
                                                                   [0, 0, 1, 0, 0, 0],
                                                                   [0, 0, 0, 1, 0, 0]]], dtype=dtype, device=device)

        # Just SP inverse projection
        data = torch.tensor([[0, 0, 1, 0],
                             [0.2, 0.3, 0.4, 0.1]], dtype=dtype, device=device)

        packet = InversePassOutputPacket(data, node.outputs.tp.projection_outputs)
        projected = node.recursive_inverse_projection_from_output(packet)

        # The result of the projection itself would be [[0, 0, 0.5, 0.5, 0, 0], ...], and it should be viewed as (2, 3, 2).
        expected_projection = torch.tensor([[[0, 0], [0.5, 0.5], [0, 0]],
                                            [[0.2, 0.3], [0.4, 0.1], [0, 0]]], dtype=dtype, device=device)

        assert same(expected_projection, projected[0].tensor)

        # TODO enable TP inverse projection calculation and uncomment following code

        # TP + SP inverse projection
        # node._unit.flock.tp_flock.frequent_seqs = torch.tensor([[[0, 2, 0], [0, 2, 1]], [[0, 1, 0], [2, 3, 2]]],
        #                                                        dtype=torch.int64, device=device)
        # node._unit.flock.tp_flock.frequent_seq_likelihoods_priors_clusters_context = torch.tensor(
        #     [[1.0, 1.0], [1.0, 1.0]], dtype=dtype, device=device)
        #
        # data = torch.tensor([[0.25, 0.25, 0.5, 0.0],
        #                      [0.5, 0.5, 0.0, 0.0]], dtype=dtype, device=device)
        #
        # packet = InversePassOutputPacket(data, node.outputs.tp.projection_outputs)
        # projected = node.recursive_inverse_projection_from_output(packet)
        #
        # expected_projection = torch.tensor([[[0.333, 0.333], [0.167, 0.167], [0, 0]],
        #                                     [[0.667, 0.333], [0.0, 0.0], [0, 0]]], dtype=dtype, device=device)
        #
        # assert same(expected_projection, projected[0].tensor, eps=1e-3)
