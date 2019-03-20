import torch
import numpy as np

from typing import Generator, List, Any
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.nodes.weighted_avg_node import WeightedAvgNode

from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestWeightedAvgNode(NodeTestBase):
    def setup_class(self, device: str = 'cuda'):
        super().setup_class()

    node_input = [[[1., 1., 0.],
                   [1., 1., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.]],

                  [[0., 1., 1.],
                   [0., 1., 1.],
                   [0., 0., 0.],
                   [0., 0., 0.]],

                  [[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 1., 1.],
                   [0., 1., 1.]]]

    node_weights = [.1, .3, .5]

    node_output = [[.0, .0, .0],
                   [.0, .0, .0],
                   [.0, .0, .0],
                   [.0, .0, .0]]

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        yield [self._creator.tensor(self.node_input, dtype=self._dtype, device=self._device)]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        node_output = self.node_output.copy()
        for i in range(len(self.node_weights)):
            node_output += np.dot(self.node_input[i], self.node_weights[i])

        yield [self._creator.tensor(node_output, dtype=self._dtype, device=self._device)]

    def _create_node(self) -> WorkerNodeBase:
        node = WeightedAvgNode()
        node.input_weights = self.node_weights
        node.weights_on_input = False
        return node


class TestWeightedAvgNodeWithInput(NodeTestBase):
    def setup_class(self, device: str = 'cuda'):
        super().setup_class()

    node_input = [[[1., 1., 0.],
                   [1., 1., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.]],

                  [[0., 1., 1.],
                   [0., 1., 1.],
                   [0., 0., 0.],
                   [0., 0., 0.]],

                  [[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 1., 1.],
                   [0., 1., 1.]]]

    node_weights = [.1, .3, .5]

    node_output = [[.0, .0, .0],
                   [.0, .0, .0],
                   [.0, .0, .0],
                   [.0, .0, .0]]

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        yield [self._creator.tensor(self.node_input, dtype=self._dtype, device=self._device),
               self._creator.tensor(self.node_weights, dtype=self._dtype, device=self._device)]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        node_output = self.node_output.copy()
        for i in range(len(self.node_weights)):
            node_output += np.dot(self.node_input[i], self.node_weights[i])

        yield [self._creator.tensor(node_output, dtype=self._dtype, device=self._device)]

    def _create_node(self) -> WorkerNodeBase:
        node = WeightedAvgNode()
        node.weights_on_input = True
        return node


class TestWeightedAvgNodeWithInputUnused(NodeTestBase):
    def setup_class(self, device: str = 'cuda'):
        super().setup_class()

    node_input = [[[1., 1., 0.],
                   [1., 1., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.]],

                  [[0., 1., 1.],
                   [0., 1., 1.],
                   [0., 0., 0.],
                   [0., 0., 0.]],

                  [[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 1., 1.],
                   [0., 1., 1.]]]

    node_weights = [.1, .3, .5]

    node_output = [[.0, .0, .0],
                   [.0, .0, .0],
                   [.0, .0, .0],
                   [.0, .0, .0]]

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        yield [self._creator.tensor(self.node_input, dtype=self._dtype, device=self._device),
               self._creator.tensor(self.node_weights, dtype=self._dtype, device=self._device)]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        node_output = self.node_output.copy()
        for i in range(len(self.node_weights)):
            node_output += np.dot(self.node_input[i], self.node_weights[i])

        yield [self._creator.tensor(node_output, dtype=self._dtype, device=self._device)]

    def _create_node(self) -> WorkerNodeBase:
        node = WeightedAvgNode()
        node.input_weights = self.node_weights
        node.weights_on_input = False
        return node
