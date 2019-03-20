import pytest
import torch

from torchsim.core.graph.connection import Connector
from torchsim.core.nodes.switch_node import SwitchUnit, SwitchNode
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestSwitch(NodeTestBase):
    _shape: torch.Size = torch.Size((1, 2))

    def _generate_input_tensors(self):
        yield [torch.zeros(self._shape, device=self._device, dtype=self._dtype),
               torch.ones(self._shape, device=self._device, dtype=self._dtype)]

    def _generate_expected_results(self):
        yield [torch.ones(self._shape, device=self._device, dtype=self._dtype)]

    def _create_node(self):
        return SwitchNode(n_inputs=2, active_input_index=1)


class TestSwitchIndexOnInput(NodeTestBase):
    _shape: torch.Size = torch.Size((1, 2))

    def _generate_input_tensors(self):
        yield [torch.zeros(self._shape, device=self._device, dtype=self._dtype),
               torch.ones(self._shape, device=self._device, dtype=self._dtype),
               torch.tensor([1], device=self._device, dtype=self._dtype)]

    def _generate_expected_results(self):
        yield [torch.ones(self._shape, device=self._device, dtype=self._dtype)]

    def _create_node(self):
        return SwitchNode(n_inputs=2, get_index_from_input=True, active_input_index=0)


class TestSwitchIndexOnInputVector(NodeTestBase):
    _shape: torch.Size = torch.Size((1, 2))

    def _generate_input_tensors(self):
        yield [torch.zeros(self._shape, device=self._device, dtype=self._dtype),
               torch.ones(self._shape, device=self._device, dtype=self._dtype),
               torch.tensor([0, 1], device=self._device, dtype=self._dtype)]

    def _generate_expected_results(self):
        yield [torch.ones(self._shape, device=self._device, dtype=self._dtype)]

    def _create_node(self):
        return SwitchNode(n_inputs=2, get_index_from_input=True, active_input_index=0)
