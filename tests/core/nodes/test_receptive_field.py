from typing import Generator, List, Any

import pytest
import torch

from torchsim.core import get_float
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.invertible_node import InversePassOutputPacket
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.nodes.receptive_field_node import ReceptiveFieldUnit, ReceptiveFieldNode
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.receptive_field.grid import Stride
from torchsim.utils.param_utils import Size2D
from torchsim.core.utils.tensor_utils import same
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestReceptiveFieldUnit:
    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_rf_unit(self, device):
        float_dtype = get_float(device)
        parent_rf_size_x = parent_rf_size_y = 4
        n_channels = 4
        image_grid_size_x = image_grid_size_y = 16
        dimensions = (image_grid_size_y, image_grid_size_x, n_channels)
        parent_rf_dims = Size2D(parent_rf_size_y, parent_rf_size_x)
        unit = ReceptiveFieldUnit(AllocatingCreator(device), dimensions, parent_rf_dims,
                                  flatten_output_grid_dimensions=True)

        input_image = torch.zeros(image_grid_size_y, image_grid_size_x, n_channels, dtype=float_dtype, device=device)
        input_image[0, parent_rf_size_x, 0] = 1

        unit.step(input_image)
        node_output = unit.output_tensor

        n_parent_rfs = (image_grid_size_x // parent_rf_size_x) * (image_grid_size_y // parent_rf_size_y)
        assert node_output.shape == torch.Size([n_parent_rfs, parent_rf_size_y, parent_rf_size_x, n_channels])
        assert node_output[1, 0, 0, 0] == 1
        # assert node_output.interpret_shape == [n_parent_rfs, parent_rf_size_y, parent_rf_size_x, n_channels]

        back_projection = unit.inverse_projection(node_output)
        # assert back_projection.interpret_shape == input_image.shape
        assert back_projection.equal(input_image)


class TestReceptiveFieldNodeDirect:
    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_rf_node(self, device):
        float_dtype = get_float(device)
        parent_rf_size_x = parent_rf_size_y = 4
        n_channels = 4
        image_grid_size_x = image_grid_size_y = 16
        dimensions = (image_grid_size_y, image_grid_size_x, n_channels)
        parent_rf_dims = Size2D(parent_rf_size_y, parent_rf_size_x)

        graph = Topology(device)

        node = ReceptiveFieldNode(dimensions, parent_rf_dims, flatten_output_grid_dimensions=True)

        graph.add_node(node)

        memory_block = MemoryBlock()
        memory_block.tensor = torch.zeros(image_grid_size_y, image_grid_size_x, n_channels, dtype=float_dtype,
                                          device=device)
        memory_block.tensor[0, parent_rf_size_x, 0] = 1

        Connector.connect(memory_block, node.inputs.input)

        graph.prepare()

        graph.step()

        node_output = node.outputs.output.tensor

        n_parent_rfs = (image_grid_size_y // parent_rf_size_y) * (image_grid_size_x // parent_rf_size_x)
        assert node_output.shape == torch.Size([n_parent_rfs, parent_rf_size_y, parent_rf_size_x, n_channels])
        assert node_output[1, 0, 0, 0] == 1

        back_projection = node.recursive_inverse_projection_from_output(InversePassOutputPacket(node_output,
                                                                                                node.outputs.output))
        # assert back_projection.interpret_shape == input_image.shape
        assert same(back_projection[0].tensor, memory_block.tensor)

    def test_expert_dimensions(self):
        """Tests multi-dimensional expert indexes."""
        device = 'cpu'
        parent_rf_size_x = parent_rf_size_y = 4
        n_channels = 4
        image_grid_size_x = image_grid_size_y = 16
        input_dimensions = (image_grid_size_y, image_grid_size_x, n_channels)
        parent_rf_dims = Size2D(parent_rf_size_x, parent_rf_size_y)
        parent_grid_dimensions = (4, 4)

        graph = Topology(device)

        node = ReceptiveFieldNode(input_dimensions, parent_rf_dims)

        graph.add_node(node)

        memory_block = MemoryBlock()
        memory_block.tensor = torch.zeros(image_grid_size_y, image_grid_size_x, n_channels, device=device)
        memory_block.tensor[0, parent_rf_size_x, 0] = 1

        Connector.connect(memory_block, node.inputs.input)

        graph.prepare()

        graph.step()

        node_output = node.outputs.output.tensor

        assert node_output.shape == torch.Size(
            parent_grid_dimensions + (parent_rf_size_y, parent_rf_size_x, n_channels))
        assert node_output[0, 1, 0, 0, 0] == 1


class TestReceptiveFieldNode(NodeTestBase):
    """Test node + serialization."""

    @classmethod
    def setup_class(cls, device: str = 'cpu'):
        super().setup_class(device)
        cls.float_dtype = get_float(cls._device)
        cls.parent_rf_size_x = 2
        cls.parent_rf_size_y = 3
        cls.n_channels = 5
        cls.image_grid_size_x = 4
        cls.image_grid_size_y = 6
        cls.dimensions = (cls.image_grid_size_y, cls.image_grid_size_x, cls.n_channels)
        cls.parent_rf_dims = Size2D(cls.parent_rf_size_y, cls.parent_rf_size_x)
        cls.parent_rf_stride_dims = Stride(3, 1)

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        input = self._creator.zeros(self.image_grid_size_y, self.image_grid_size_x, self.n_channels,
                                    dtype=self.float_dtype)
        input[2, 3, 4] = -3.14
        input[0, 0, 0] = 123
        input[5, 3, 3] = 456

        yield [input]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        parent_rfs = (self.image_grid_size_y // self.parent_rf_size_y,
                      3)
        output = self._creator.zeros((parent_rfs[0], parent_rfs[1], self.parent_rf_size_y, self.parent_rf_size_x,
                                      self.n_channels), dtype=self.float_dtype)
        output[0, 2, 2, 1, 4] = -3.14
        output[0, 0, 0, 0, 0] = 123
        output[1, 2, 2, 1, 3] = 456

        yield [output]

    def _create_node(self) -> ReceptiveFieldNode:
        return ReceptiveFieldNode(input_dims=self.dimensions, parent_rf_dims=self.parent_rf_dims,
                                  parent_rf_stride=self.parent_rf_stride_dims,
                                  flatten_output_grid_dimensions=False)
