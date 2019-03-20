from dataclasses import dataclass
from typing import Tuple

import torch
from torchsim.core.graph.inverse_pass_packet import InversePassInputPacket
from torchsim.core.graph.invertible_node import InversePassOutputPacket
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.graph.worker_node_base import InvertibleWorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.models.receptive_field.grid import Grids, Stride
from torchsim.core.models.receptive_field.mapping import Mapping
from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime
from torchsim.gui.validators import *
from torchsim.utils.param_utils import Size2D
from torch import Tensor


class ReceptiveFieldUnit(InvertibleUnit):
    """Performs mapping of output from a child-level receptive field grid to parent expert input."""

    _parent_rf_dims: Size2D
    _parent_rf_stride_dims: Stride
    _rf_mapping: Mapping = None

    def __init__(self,
                 creator: TensorCreator,
                 input_dims: Tuple[int, int, int],
                 parent_rf_dims: Size2D,
                 parent_rf_stride_dims: Stride = None,
                 flatten_output_grid_dimensions=False):
        super().__init__(creator.device)
        self._parent_rf_dims = parent_rf_dims
        self._parent_rf_stride_dims = parent_rf_stride_dims
        self.creator = creator
        self._rf_mapping = self.create_rf_mapping(input_dims,
                                                  parent_rf_dims,
                                                  parent_rf_stride_dims,
                                                  flatten_output_grid_dimensions,
                                                  self._device)
        output_dims = tuple(
            self._rf_mapping(torch.zeros(input_dims, device=self._device, dtype=self._float_dtype)).shape)
        self.output_tensor = creator.zeros(output_dims, device=self._device, dtype=self._float_dtype)

    def step(self, data: torch.Tensor):
        self.output_tensor.copy_(self._rf_mapping(data))

    def inverse_projection(self, node_output: torch.Tensor) -> torch.Tensor:
        """Recovers the input (or one possible input) from node output.

        This is the inversion of the node forward pass.
        """
        return self._rf_mapping.inverse_map(node_output)

    @staticmethod
    def create_rf_mapping(input_dims: Tuple[int, int, int], parent_rf_dims, parent_rf_stride_dims,
                          flatten_output_grid_dimensions, device):
        bottom_grid_y, bottom_grid_x, chunk_size = input_dims
        grids = Grids(Size2D(bottom_grid_y, bottom_grid_x), parent_rf_dims, parent_rf_stride=parent_rf_stride_dims,
                      flatten_output_grid_dimensions=flatten_output_grid_dimensions)
        return Mapping.from_default_input(grids, device, chunk_size)


class ReceptiveFieldInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("ReceptiveFieldInput")


class ReceptiveFieldOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("ReceptiveFieldOutput")

    def prepare_slots(self, unit: ReceptiveFieldUnit):
        self.output.tensor = unit.output_tensor


@dataclass
class ReceptiveFieldNodeParams(ParamsBase):
    input_dims: Tuple[int, int, int]
    parent_rf_dims: Size2D
    parent_rf_stride: Stride
    flatten_output_grid_dimensions: bool


class ReceptiveFieldNode(InvertibleWorkerNodeBase):
    """Treats its input as a grid of local receptive fields and maps it to parent expert input.

    The node is useful when we want to treat data as composed of smaller local receptive fields.
    In this case, we need to divide the input into a grid of local receptive fields (LRFs) and rearrange the tensor
    so that the data contained in each LRF is contiguous. This makes it possible for a separate expert to
    process each LRF.

    The node is initialized with the input dimensions and the dimensions of a local receptive field.
    There is an optional stride parameter to support overlapping receptive fields.

    The inputs of the node has the dimensions (Y, X, C) corresponding to the height, width, and channels of the input.
    Outputs have dimensions (Y, X, Y_rf, X_rf, C), where Y, X are the numbers of output receptive fields (experts)
    in each dimension and Y_rf, X_rf, are the height and width of an RF. If flatten_output_grid_dimensions is True,
    the outputs have dimensions (N_rfs, Y_rf, X_rf, C), where N_rfs is the number of output receptive fields.
    """
    _unit: ReceptiveFieldUnit
    inputs: ReceptiveFieldInputs
    outputs: ReceptiveFieldOutputs
    _params: ReceptiveFieldNodeParams

    def __init__(self,
                 input_dims: Tuple[int, int, int],
                 parent_rf_dims: Size2D,
                 parent_rf_stride: Optional[Stride] = None,
                 flatten_output_grid_dimensions=False,
                 name="ReceptiveField"):
        """Initializes the receptive field node.

        Args:
            input_dims: The input dimensions in YXC format, (32, 64, 3) for an RGB image with height=32, width=64
            parent_rf_dims: (height, width) of a single LRF
            parent_rf_stride: (y, x) stride (None if no overlap or gap between LRFs)
            flatten_output_grid_dimensions: If true, output rfs are accessed through a 1D index
            name: node name
        """
        super().__init__(name=name, inputs=ReceptiveFieldInputs(self),
                         outputs=ReceptiveFieldOutputs(self))
        self._params = ReceptiveFieldNodeParams(
            input_dims=input_dims,
            parent_rf_dims=parent_rf_dims,
            parent_rf_stride=parent_rf_stride or Stride(*parent_rf_dims),
            flatten_output_grid_dimensions=flatten_output_grid_dimensions
        )

    def _create_unit(self, creator: TensorCreator) -> ReceptiveFieldUnit:
        self._derive_params()
        return ReceptiveFieldUnit(creator,
                                  parent_rf_dims=self._params.parent_rf_dims,
                                  parent_rf_stride_dims=self._params.parent_rf_stride,
                                  input_dims=self._params.input_dims,
                                  flatten_output_grid_dimensions=self._params.flatten_output_grid_dimensions)

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        projection = self._unit.inverse_projection(data.tensor.view(self._unit.output_tensor.shape))
        return [InversePassInputPacket(projection.view(self.inputs.input.tensor.shape), self.inputs.input)]

    def inverse_projection(self, data: Tensor) -> Tensor:
        return self._unit.inverse_projection(data)

    def validate(self):
        if self.inputs.input.tensor.dim() == 2:
            assert self._params.input_dims == self.inputs.input.tensor.unsqueeze(2).shape
        else:
            assert self._params.input_dims == self.inputs.input.tensor.shape

    def _step(self):
        self._unit.step(self.inputs.input.tensor)

    def _derive_params(self):
        """Derive the params of the node from the input shape."""
        self._params.input_dims = self.inputs.input.tensor.shape

        # input shape could be (Y, X) without the channel, add the last dimension
        if len(self._params.input_dims) == 2:
            self._params.input_dims = (*self._params.input_dims, 1)

    @property
    def input_dims(self) -> List[int]:
        return list(self._params.input_dims)

    @input_dims.setter
    def input_dims(self, value: List[int]):
        self._params.input_dims = tuple(value)

    @property
    def parent_rf_dims(self) -> List[int]:
        return list(self._params.parent_rf_dims)

    @parent_rf_dims.setter
    def parent_rf_dims(self, value: List[int]):
        self._params.parent_rf_dims = tuple(value)

    @property
    def parent_rf_stride(self) -> List[int]:
        return list(self._params.parent_rf_stride)

    @parent_rf_stride.setter
    def parent_rf_stride(self, value: List[int]):
        self._params.parent_rf_stride = tuple(value)

    @property
    def flatten_output_grid_dimensions(self) -> bool:
        return self._params.flatten_output_grid_dimensions

    @flatten_output_grid_dimensions.setter
    def flatten_output_grid_dimensions(self, value: bool):
        self._params.flatten_output_grid_dimensions = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto("Input dims", type(self).input_dims, edit_strategy=disable_on_runtime),
            self._prop_builder.auto("Parent RF dims", type(self).parent_rf_dims, edit_strategy=disable_on_runtime),
            self._prop_builder.auto("Stride", type(self).parent_rf_stride, edit_strategy=disable_on_runtime),
            self._prop_builder.auto("Flatten output grid", type(self).flatten_output_grid_dimensions,
                                    edit_strategy=disable_on_runtime)
        ]
