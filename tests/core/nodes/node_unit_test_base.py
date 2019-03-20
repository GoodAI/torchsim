import logging
import tempfile
from abc import ABC, abstractmethod
from typing import List, Any, Generator

import pytest

import torch
from torchsim.core import get_float
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import NodeBase
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.core.utils.tensor_utils import same

logger = logging.getLogger(__name__)


class AnyResult:
    """A class representing a result in which its actual value does not matter.
    Used for example in a case where the node would be returning random numbers for the first four steps and then some
    value. We do not care which particular numbers it returns, so we use AnyResult as the expected result for those
    four steps and then check for the actual value we care about."""


class NodeTestBase(ABC):
    """A class for simplifying of writing node tests.

    It tests that tha node:
        - can be initialized
        - computes correct outputs in n-steps
        - serializes correctly (tested weakly by checking that the computed values after n steps are correct even after
        saving and loading
        - its properties can be gotten and set
    """

    _device: str
    _creator: AllocatingCreator
    _dtype: torch.dtype

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        cls._device = device
        cls._creator = AllocatingCreator(device=cls._device)
        cls._dtype = get_float(cls._device)

    def test_node(self):
        """
        Tests if the node makes step(s) and returns the expected values.
        """
        node, sources = self._prepare_node()
        self._run_node_for_steps(node, sources)

    @staticmethod
    def skip_test_serialization():
        """use to skip the test if it does not make sense to be tested."""
        return False

    def test_serialization(self):
        if self.skip_test_serialization():
            pytest.skip()

        """Test that if the node is saved, changed and then loaded, it still computes the expected result."""
        node, sources = self._prepare_node()

        # save the node state after initialization
        with tempfile.TemporaryDirectory() as directory:
            saver = Saver(directory)
            node.save(saver)
            saver.save()

            # run steps to change it
            self._run_node_for_steps(node, sources, check_results=False)

            self._change_node_before_load(node)

            # load the initial state
            loader = Loader(directory)
            node.load(loader)

        # run steps again and check they still produce the desired results
        self._run_node_for_steps(node, sources)

    def test_properties(self):
        """
        Tests if getter and setter of all properties works.
        """

        node, sources = self._prepare_node()

        properties = node.get_properties()

        for prop in properties:
            gotten_value = prop.value

            prop.callback(str(gotten_value))

            # Check at least that the setter did not store a different value
            assert gotten_value == prop.value

    @staticmethod
    def _inputs_to_sources(inputs):
        outputs = [MemoryBlock(None, f'output{i}') for i in range(len(inputs))]
        for output, tensor in zip(outputs, inputs):
            output.tensor = tensor

        return outputs

    @staticmethod
    def _replace_data_in_inputs(sources, input_tensors):
        """Replace tensors in source memory blocks with `input_tensors`."""
        for source, input_tensor in zip(sources, input_tensors):
            source.tensor = input_tensor

    @abstractmethod
    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        """Generator generating a list of input tensors for each step.

        The node will run for multiple steps, presenting the inputs sequentially.
        """
        pass

    @abstractmethod
    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        """Generator generating a list of expected results for each step.

        The node will run for multiple steps, presenting the inputs sequentially and checking outputs every step.
        An expected result can be the AnyResult class, in which case it will not be checked.
        """
        pass

    @abstractmethod
    def _create_node(self) -> NodeBase:
        """Return the node initialized with desired params."""
        pass

    def _extract_results(self, node) -> List[Any]:
        """Extracts results from the tested node which will be compared to expected results."""
        return [output.tensor for output in node.outputs]

    def _prepare_node(self):
        inputs = self._generate_input_tensors().__next__()
        sources = self._inputs_to_sources(inputs)

        node = self._create_node()
        for source, input_block in zip(sources, node.inputs):
            Connector.connect(source, input_block)
        node.allocate_memory_blocks(self._creator)
        node.validate()

        return node, sources

    def _run_node_for_steps(self, node, sources, check_results: bool = True):
        for step, (inputs, expected_results) in enumerate(
                zip(self._generate_input_tensors(), self._generate_expected_results())):
            self._replace_data_in_inputs(sources, inputs)
            node.step()
            if check_results:
                results = self._extract_results(node)
                self._check_results(expected_results, results, step)

    def _change_node_before_load(self, node):
        pass

    def _check_results(self, expected, results, step: int):
        for expected_tensor, result_tensor in zip(expected, results):
            assert self._same(expected_tensor, result_tensor), f"Assertion failed in step {step}."

    @staticmethod
    def _same(expected, result, eps=None) -> bool:
        if expected is AnyResult:
            return True
        elif isinstance(expected, torch.Tensor):
            return same(expected, result, eps=eps)
        else:
            return expected == result
