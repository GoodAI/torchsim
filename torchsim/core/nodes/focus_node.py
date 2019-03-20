import math
from dataclasses import dataclass
from typing import List, Optional

from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime
from torchsim.gui.validators import validate_predicate, validate_positive_with_zero_int
from torch import FloatTensor


@dataclass
class FocusNodeParams(ParamsBase):
    trim_output: bool = False
    trim_output_size: int = 0


class FocusNodeUnit(Unit):
    _width: int
    _height: int
    _num_channels: int
    _creator: TensorCreator
    _params: FocusNodeParams

    def __init__(self, creator: TensorCreator,
                 input_height: int, input_width: int,
                 output_height: int, output_width: int,
                 num_channels: int, params: FocusNodeParams):
        super().__init__(creator.device)

        self._width = output_width
        self._height = output_height
        self._num_channels = num_channels
        self._params = params
        self._creator = creator

        self.last_mask = creator.zeros([input_height, input_width, num_channels], dtype=creator.float,
                                       device=creator.device)
        self.last_output = creator.zeros([output_height, output_width, num_channels], dtype=creator.float,
                                         device=creator.device)

    def step(self, input: FloatTensor, coordinates: FloatTensor):
        focus_coords = coordinates.tolist()

        f_width = int(focus_coords[3])
        f_height = int(focus_coords[2])

        fx = int(focus_coords[1])
        fy = int(focus_coords[0])
        fw = fx + f_width
        fh = fy + f_height

        self.last_mask.zero_()
        self.last_output.zero_()

        self.last_mask[fy:fh, fx:fw] = 1

        if self._params.trim_output:
            if fw - fx > self._height and fh - fy > self._width:
                raise Exception('Focus too large!')

            self.last_output.copy_(input[fy:fh, fx:fw])
        else:
            cy = math.ceil((self._width - f_width) / 2)
            cx = math.ceil((self._height - f_height) / 2)
            self.last_output[cy:cy + f_height, cx:cx + f_width].copy_(input[fy:fh, fx:fw])


class FocusNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.focus_mask = self.create("Focus_mask")
        self.focus_output = self.create("Field_of_focus")

    def prepare_slots(self, unit: FocusNodeUnit):
        self.focus_mask.tensor = unit.last_mask
        self.focus_output.tensor = unit.last_output


class FocusNodeInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input_image = self.create("Input_image")
        self.coordinates = self.create("Focus_coordinates")


class FocusNode(WorkerNodeBase):
    """Separates where from what.

        Inputs: Input image; y, x, height, width
        Outputs: Mask (same dimensions as input image); Centered, unscaled FOF (same dimensions as input image)
        """

    _unit: FocusNodeUnit
    _params: FocusNodeParams
    inputs: FocusNodeInputs
    outputs: FocusNodeOutputs

    @property
    def trim_output(self) -> bool:
        return self._params.trim_output

    @trim_output.setter
    def trim_output(self, value: bool):
        self._params.trim_output = value

    @property
    def trim_output_size(self) -> int:
        return self._params.trim_output_size

    @trim_output_size.setter
    def trim_output_size(self, value: int):
        validate_positive_with_zero_int(value)
        self._params.trim_output_size = value

    def __init__(self, name="FocusNode", params: Optional[FocusNodeParams] = None):
        super().__init__(name=name,
                         inputs=FocusNodeInputs(self),
                         outputs=FocusNodeOutputs(self))

        self._params = params.clone() if params else FocusNodeParams()

    def _create_unit(self, creator: TensorCreator):
        input_height, input_width, n_input_channels = self.inputs.input_image.tensor.shape

        if self._params.trim_output:
            output_height, output_width = self.trim_output_size, self.trim_output_size
        else:
            output_height, output_width = input_height, input_width

        return FocusNodeUnit(creator, input_height=input_height, input_width=input_width,
                             output_height=output_height, output_width=output_width, 
                             num_channels=n_input_channels, params=self._params)

    def _step(self):
        self._unit.step(self.inputs.input_image.tensor, self.inputs.coordinates.tensor)

    def validate(self):
        """Checks that the output image and mask has correct dimensions."""
        super().validate()

        image = self.inputs.input_image.tensor
        coords = self.inputs.coordinates.tensor

        validate_predicate(lambda: image.dim() == 3,
                           f"The input should be 3D (y, x, channels) but has shape {image.shape}")

        validate_predicate(lambda: coords.dim() == 1,
                           f"The coordinates should be 1D but has shape {coords.shape}")

        validate_predicate(lambda: len(coords) == 4,
                           f"The coordinates should contain 4 values but has {len(coords)}")

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""

        return [self._prop_builder.auto('Trim output', type(self).trim_output, edit_strategy=disable_on_runtime),
                self._prop_builder.auto('Trim output size', type(self).trim_output_size,
                                        edit_strategy=disable_on_runtime)
                ]
