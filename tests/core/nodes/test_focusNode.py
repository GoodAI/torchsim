import torch
import numpy as np

from typing import Generator, List, Any
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.nodes.focus_node import FocusNode
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestFocusNode(NodeTestBase):
    def setup_class(cls, device: str='cuda'):
        super().setup_class()

    input_image = [[1, 0, 0, 0],
                   [0, 0, 1, 2],
                   [0, 0, 3, 4],
                   [0, 0, 5, 6]]

    coordinates = [1, 2, 3, 2]

    mask_image = [[0, 0, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 0, 1, 1]]

    output_image = [[0, 0, 0, 0],
                    [0, 1, 2, 0],
                    [0, 3, 4, 0],
                    [0, 5, 6, 0]]

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        yield [self._creator.tensor(self.input_image, dtype=self._dtype, device=self._device).unsqueeze(-1),
               self._creator.tensor(self.coordinates, dtype=self._dtype, device=self._device)]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        yield [self._creator.tensor(self.mask_image, dtype=self._dtype, device=self._device).unsqueeze(-1),
               self._creator.tensor(self.output_image, dtype=self._dtype, device=self._device).unsqueeze(-1)]

    def _create_node(self) -> WorkerNodeBase:
        node = FocusNode()
        node.trim_output = False
        return node


class TestFocusNode3d(NodeTestBase):
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class()

    image = [[1, 0, 0, 0],
             [0, 0, 1, 2],
             [0, 0, 3, 4],
             [0, 0, 5, 6],
             [0, 0, 15, 70]]

    input_image3d = [np.dot(image, 0.1),
                     np.dot(image, 0.2),
                     np.dot(image, 0.3)]

    coordinates = [1, 2, 3, 2]

    mask_image = [[0, 0, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 0, 0, 0]]

    mask_image3d = [mask_image, mask_image, mask_image]

    output_image = [[0, 0, 0, 0],
                    [0, 1, 2, 0],
                    [0, 3, 4, 0],
                    [0, 5, 6, 0],
                    [0, 0, 0, 0]]

    output_image3d = [np.dot(output_image, .1),
                      np.dot(output_image, .2),
                      np.dot(output_image, .3)]

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        yield [self._creator.tensor(self.input_image3d, dtype=self._dtype, device=self._device).permute(1, 2, 0),
               self._creator.tensor(self.coordinates, dtype=self._dtype, device=self._device)]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        yield [self._creator.tensor(self.mask_image3d, dtype=self._dtype, device=self._device).permute(1, 2, 0),
               self._creator.tensor(self.output_image3d, dtype=self._dtype, device=self._device).permute(1, 2, 0)]

    def _create_node(self) -> WorkerNodeBase:
        node = FocusNode()
        node.trim_output = False
        return node


class TestFocusNodeTrim(NodeTestBase):
    def setup_class(cls, device: str='cuda'):
        super().setup_class()

    input_image = [[1, 0, 0, 0],
                   [0, 0, 1, 2],
                   [0, 0, 3, 4],
                   [0, 0, 5, 6]]

    coordinates = [1, 2, 2, 2]

    mask_image = [[0, 0, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 0, 0, 0]]

    output_image = [[1, 2],
                    [3, 4]]

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        yield [self._creator.tensor(self.input_image, dtype=self._dtype, device=self._device).unsqueeze(-1),
               self._creator.tensor(self.coordinates, dtype=self._dtype, device=self._device)]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        yield [self._creator.tensor(self.mask_image, dtype=self._dtype, device=self._device).unsqueeze(-1),
               self._creator.tensor(self.output_image, dtype=self._dtype, device=self._device).unsqueeze(-1)]

    def _create_node(self) -> WorkerNodeBase:
        node = FocusNode()
        node.trim_output = True
        node.trim_output_size = 2
        return node
