from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.models.receptive_field.grid import Stride, Grids
from torchsim.core.models.receptive_field.reverse_mapping import ReverseMapping
from torchsim.core.nodes.receptive_field_node import ReceptiveFieldUnit
from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime
from torchsim.utils.param_utils import Size2D


class ReceptiveFieldReverseUnit(Unit):

    def __init__(self,
                 creator: TensorCreator,
                 data_size: Tuple[int, ...],
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
        self.output_tensor = creator.zeros(data_size, device=self._device, dtype=self._float_dtype)

    def step(self, data: torch.Tensor):
        self.output_tensor.copy_(self._rf_mapping.reverse_map_concat(data))

    @staticmethod
    def create_rf_mapping(input_dims: Tuple[int, int, int], parent_rf_dims, parent_rf_stride_dims,
                          flatten_output_grid_dimensions, device):
        bottom_grid_y, bottom_grid_x, chunk_size = input_dims
        grids = Grids(Size2D(bottom_grid_y, bottom_grid_x), parent_rf_dims, parent_rf_stride=parent_rf_stride_dims,
                      flatten_output_grid_dimensions=flatten_output_grid_dimensions)
        return ReverseMapping(grids, device, chunk_size)


class ReceptiveFieldReverseInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("ReceptiveFieldInput")


class ReceptiveFieldReverseOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("ReceptiveFieldOutput")

    def prepare_slots(self, unit: ReceptiveFieldUnit):
        self.output.tensor = unit.output_tensor


@dataclass
class ReceptiveFieldReverseNodeParams(ParamsBase):
    data_size: Tuple[int, ...]
    input_dims: Tuple[int, int, int]
    parent_rf_dims: Size2D
    parent_rf_stride: Stride
    flatten_output_grid_dimensions: bool


class ReceptiveFieldReverseNode(WorkerNodeBase):
    """Reverse transformation for ReceptiveFieldNode.

    See Also: ReceptiveFieldNode for detailed description of parameters
    """
    _unit: ReceptiveFieldUnit
    inputs: ReceptiveFieldReverseInputs
    outputs: ReceptiveFieldReverseOutputs
    _params: ReceptiveFieldReverseNodeParams

    def __init__(self,
                 # Data size that will be received. TODO replace this with automatic dimension negotiation
                 data_size: Tuple[int, ...],
                 input_dims: Tuple[int, int, int],
                 parent_rf_dims: Size2D,
                 parent_rf_stride: Stride = None,
                 flatten_output_grid_dimensions=False,
                 name="ReceptiveFieldContext"):
        """Initializes the receptive field node.

        Args:
            data_size: Size of data that will be connected to input
            input_dims: The input dimensions in YXC format, (32, 64, 3) for an RGB image with height=32, width=64
            parent_rf_dims: (height, width) of a single LRF
            parent_rf_stride: (y, x) stride (None if no overlap or gap between LRFs)
            flatten_output_grid_dimensions: If true, output rfs are accessed through a 1D index
            name: node name
        """
        super().__init__(name=name, inputs=ReceptiveFieldReverseInputs(self),
                         outputs=ReceptiveFieldReverseOutputs(self
                                                              # , ReceptiveFieldUnit.output_dims(
                                                              #     input_dims,
                                                              #     parent_rf_dims,
                                                              #     'cpu'
                                                              #     )
                                                              ))
        self._params = ReceptiveFieldReverseNodeParams(
            data_size=data_size,
            input_dims=input_dims,
            parent_rf_dims=parent_rf_dims,
            parent_rf_stride=parent_rf_stride,
            flatten_output_grid_dimensions=flatten_output_grid_dimensions
        )

    def _create_unit(self, creator: TensorCreator) -> ReceptiveFieldReverseUnit:
        return ReceptiveFieldReverseUnit(creator,
                                         data_size=self._params.data_size,
                                         parent_rf_dims=self._params.parent_rf_dims,
                                         parent_rf_stride_dims=self._params.parent_rf_stride,
                                         input_dims=self._params.input_dims,
                                         flatten_output_grid_dimensions=self._params.flatten_output_grid_dimensions)

    def _step(self):
        self._unit.step(self.inputs.input.tensor)

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
    def parent_rf_dims(self, value: Optional[List[int]]):
        self._params.parent_rf_dims = tuple(value)

    @property
    def parent_rf_stride(self) -> Optional[List[int]]:
        return None if self._params.parent_rf_stride is None else list(self._params.parent_rf_stride)

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
