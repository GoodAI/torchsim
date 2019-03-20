from typing import Tuple, List

import torch

from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.core.utils.image_processing_utilities import ImageProcessingUtilities
from torchsim.gui.observables import ObserverPropertiesItem, enable_on_runtime
from torchsim.gui.validators import validate_predicate, validate_float_in_range
from torch import FloatTensor


class MotionDetectionParams(ParamsBase):

    use_thresholding: bool = False  # threshold the motion_map output to binary values?
    threshold_value: float = 0.025  # according to what threshold to binarize (if enabled)


class MotionDetectionUnit(Unit):
    _height: int
    _width: int
    _num_channels: int
    _params: MotionDetectionParams

    _has_previous_image: bool

    def __init__(self,
                 creator: TensorCreator,
                 height: int,
                 width: int,
                 num_channels: int,
                 params: MotionDetectionParams):
        super().__init__(creator.device)

        self._height = height
        self._width = width
        self._num_channels = num_channels

        self._params = params

        self._has_previous_image = False

        bw_shape = [self._height, self._width]

        self.previous_image = creator.zeros(bw_shape, dtype=creator.float, device=creator.device)
        self.temp_image = creator.zeros(bw_shape, dtype=creator.float, device=creator.device)
        self.motion_map = creator.zeros(bw_shape, dtype=creator.float, device=creator.device)

    def step(self, input_image: FloatTensor):

        # properly convert to grayscale if necessary
        if self._num_channels == 1:
            input = input_image.squeeze(-1)
        else:
            input = ImageProcessingUtilities.rgb_to_grayscale(input_image, True)

        if not self._has_previous_image:
            self.previous_image.copy_(input)
            self._has_previous_image = True
            return

        # compute the absolute difference between consecutive images
        self.temp_image = torch.abs(input - self.previous_image)

        # threshold if needed
        if self._params.use_thresholding:
            self.motion_map.zero_()
            self.motion_map[self.temp_image > self._params.threshold_value] = 1.0
        else:
            self.motion_map.copy_(self.temp_image)

        # store the previous image
        self.previous_image.copy_(input)

    def _save(self, saver: Saver):
        saver.description['_has_previous_image'] = self._has_previous_image

    def _load(self, loader: Loader):
        self._has_previous_image = loader.description['_has_previous_image']


class MotionDetectionNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.motion_map = self.create("Motion_map")

    def prepare_slots(self, unit: MotionDetectionUnit):
        self.motion_map.tensor = unit.motion_map


class MotionDetectionNodeInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input_image = self.create("Input_image")


class MotionDetectionNode(WorkerNodeBase):
    """ Given the two consecutive images, compute motion_map which represents parts of the scene that moved.

    Currently just a absolute pixel difference is used for computing the motion_map.

    Inputs: Input image [Y, X, n_channels]
    Outputs: motion map [Y, X]
    """

    _params: MotionDetectionParams

    _unit: MotionDetectionUnit
    inputs: MotionDetectionNodeInputs
    outputs: MotionDetectionNodeOutputs

    _image_shape: Tuple[int, int, int]

    _image_height: int
    _image_width: int
    _image_num_channels: int

    def __init__(self,
                 name="MotionDetectionNode",
                 params: MotionDetectionParams = MotionDetectionParams()):
        super().__init__(name=name,
                         inputs=MotionDetectionNodeInputs(self),
                         outputs=MotionDetectionNodeOutputs(self))

        self._params = params.clone()

    def _create_unit(self, creator: TensorCreator):
        self._derive_params()

        return MotionDetectionUnit(creator,
                                   height=self._image_height,
                                   width=self._image_width,
                                   num_channels=self._image_num_channels,
                                   params=self._params)

    def _step(self):
        self._unit.step(self.inputs.input_image.tensor)

    def validate(self):
        super().validate()
        shape = self.inputs.input_image.tensor.shape
        validate_predicate(lambda: len(shape) == 3,
                           f"shape of the input should be 3 (Y,X,channel(s)), but is {shape}")
        validate_predicate(lambda: shape[2] == 3 or shape[2] == 1,
                           f"the last dimension corresponds to channels, only values 1 (grayscale) or 3 (RGB) supported")

    def _derive_params(self):
        shape = self.inputs.input_image.tensor.shape

        if len(shape) == 3:
            self._image_height = shape[0]
            self._image_width = shape[1]
            self._image_num_channels = shape[2]
        else:
            # invalid input dimensions, will not go through the validation
            self._image_height = self._image_width = self._image_num_channels = 1

    @property
    def use_thresholding(self) -> bool:
        return self._params.use_thresholding

    @use_thresholding.setter
    def use_thresholding(self, value: bool):
        self._params.use_thresholding = value
        self._unit._params.use_thresholding = value

    @property
    def threshold_value(self) -> float:
        return self._params.threshold_value

    @threshold_value.setter
    def threshold_value(self, value: float):
        validate_float_in_range(value, 0.0, 1.0)
        self._params.threshold_value = value
        self._unit._params.threshold_value = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('Use thresholding',
                                    type(self).use_thresholding,
                                    edit_strategy=enable_on_runtime),
            self._prop_builder.auto('Threshold value',
                                    type(self).threshold_value,
                                    edit_strategy=enable_on_runtime)]

