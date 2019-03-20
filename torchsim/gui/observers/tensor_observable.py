import logging
import math
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torchsim.core.global_settings import GlobalSettings
from torchsim.core.memory.tensor_creator import TensorSurrogate
from torchsim.core.model import PropertiesProvider
from torchsim.gui.observables import ObserverPropertiesItem, MemoryBlockObservable, ObserverCallbacks, \
    ObserverPropertiesBuilder
from torchsim.gui.server.ui_server_connector import RequestData
from torchsim.gui.validators import validate_dimension_vs_shape

logger = logging.getLogger(__name__)


def sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None:
        return torch.full((1,), float('nan'))
    elif type(tensor) is TensorSurrogate:
        return torch.full(tensor.size(), float('nan'))
    else:
        return tensor.clone()


@dataclass
class TensorViewProjectionUIParams:
    width: int
    height: int
    items_per_row: int
    count: int


@dataclass
class TensorObservableParams:
    scale: int
    projection: TensorViewProjectionUIParams


@dataclass
class TensorObservableData:
    tensor: torch.Tensor
    params: TensorObservableParams


def dummy_tensor_observable_data():
    return TensorObservableData(torch.tensor([[[1, 1, 0.]]]),
                                TensorObservableParams(GlobalSettings.instance().observer_memory_block_minimal_size,
                                                       TensorViewProjectionUIParams(1, 1, 1, 1)))


class TensorViewProjection(ABC, PropertiesProvider):
    _real_shape: List[int]
    _shape: List[int]
    _items_per_row: int
    _min: float
    _max: float
    _sum_dim: int
    _prop_builder: ObserverPropertiesBuilder

    @property
    def min(self) -> float:
        return self._min

    @min.setter
    def min(self, value: float):
        self._min = value

    @property
    def max(self) -> float:
        return self._max

    @max.setter
    def max(self, value: float):
        self._max = value

    @property
    def items_per_row(self) -> int:
        return self._items_per_row

    @items_per_row.setter
    def items_per_row(self, value: int):
        self._items_per_row = value

    @property
    def shape(self) -> List[int]:
        return self._shape

    @shape.setter
    def shape(self, value: List[int]):
        self._shape = value

    @property
    def real_shape(self) -> List[int]:
        return self._real_shape

    @property
    def sum_dim(self) -> Optional[int]:
        return self._sum_dim

    @sum_dim.setter
    def sum_dim(self, value: Optional[int]):
        validate_dimension_vs_shape(value, self._real_shape)
        self._sum_dim = value

    def __init__(self, is_buffer: bool):
        self._real_shape = []
        self._shape = []
        self._items_per_row = 1
        self._min = 0
        self._max = 1
        self._logger = logging.getLogger(f"{__name__}.Observer.{type(self).__name__}")
        self._is_buffer = is_buffer
        self._sum_dim = None
        self._prop_builder = ObserverPropertiesBuilder(self)

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('Min', type(self).min),
            self._prop_builder.auto('Max', type(self).max),
            self._prop_builder.auto('Items per row', type(self).items_per_row),
            self._prop_builder.auto('Shape', type(self).shape),
            self._prop_builder.auto('Real shape', type(self).real_shape),
            self._prop_builder.auto('Sum over dim', type(self).sum_dim)
        ]

    def _compute_tile_dimensions(self, size: List[int], is_rgb: bool):
        if len(self._shape) >= 2:
            return self._shape[-2:]

        # no shape defined - use last two dimensions of the tensor
        if is_rgb:
            size = size[:-1]  # drop channel dimensions

        height, width = [1, 1] if len(size) < 2 else size[-2:]
        if self._is_buffer:
            # buffers have dimensions [flock_size, buffer_size, data]
            # height is set to 1 to cover just one buffer item in the tile
            height = 1
        return height, width

    def transform_tensor(self, tensor: torch.Tensor, is_rgb: bool) -> Tuple[torch.Tensor, TensorViewProjectionUIParams]:

        tensor = self.apply_any_transforms(tensor)
        self._real_shape = list(tensor.shape)

        is_rgb_three_channels = is_rgb and (tensor.size()[-1] == 3)
        is_rgb_single_channel = is_rgb and (tensor.size()[-1] == 1)

        height, width = self._compute_tile_dimensions(list(tensor.size()),
                                                      is_rgb_single_channel or is_rgb_three_channels)
        if not is_rgb_three_channels:
            tensor = tensor.contiguous().view([-1])
            tensor = self._colorize(tensor, self._min, self._max)
        else:
            tensor = self._rgb_transform(tensor, self._min, self._max)

        # Create column vector with all 2D images from top to bottom
        return self._compose_tensor_rgb(width, height, tensor)

    def apply_any_transforms(self, tensor: torch.Tensor) -> torch.Tensor:

        # We don't compute transforms on all NaN tensors - the simulation has probably not yet started
        # noinspection PyUnresolvedReferences
        if self._sum_dim is not None and torch.isnan(tensor).all().item() == 0:
            tensor = tensor.sum(self._sum_dim)
        return tensor

    @staticmethod
    def _column_tiling_indices(device: str, count: int, width: int, height: int):
        a = torch.arange(0, count, device=device)
        a = a.view(-1, width * height)
        i = torch.arange(0, width * height, device=device)
        i = i.view(width, height).transpose(0, 1).contiguous().view(-1)
        ri = a.index_select(1, i).view(-1)
        return ri

    # def _compose_tensor_simple(self, width: int, height: int, count: int, column_tensor: torch.Tensor):
    #     result_dims = [
    #         math.ceil(count / self._items_per_row) * height,
    #         self._items_per_row * width
    #     ]
    #
    #     # Pad column tensor so it can be viewed as canvas of result_dims
    #     column_height = column_tensor.size()[0]
    #     excess_image_rows = column_height % height
    #     missing_image_rows = 0 if excess_image_rows == 0 else height - excess_image_rows
    #
    #     excess_images = count % self._items_per_row
    #     missing_images = 0 if excess_images == 0 else self._items_per_row - excess_images
    #
    #     pad_tensor = torch.zeros((missing_image_rows + missing_images * height, width), device=column_tensor.device)
    #
    #     padded_column_tensor = torch.cat([column_tensor, pad_tensor]).view([-1, width])
    #
    #     # Compute tiling indices
    #     image_rows = padded_column_tensor.size(0)
    #     indices = self._column_tiling_indices(column_tensor.device, image_rows, self._items_per_row, height)
    #
    #     # Reorder tensor in order to make tiling
    #     images = padded_column_tensor.index_select(0, indices)
    #     return images.view(result_dims), TensorViewProjectionUIParams(width, height, self._items_per_row)

    @staticmethod
    def _compute_padding(value: int, divisor: int) -> int:
        """Compute how many elements have to be added so value is divisible by divisor.

        Args:
            value:
            divisor:

        Returns:
            Number of elements that needs to be added.
        """
        excess = value % divisor
        return 0 if excess == 0 else divisor - excess

    def _compose_tensor_rgb(self, width: int, height: int, tensor: torch.Tensor):
        pad_color = torch.tensor([0.3, 0.3, 0.3], dtype=tensor.dtype, device=tensor.device)

        # if len(tensor.size()) < 3:
        #     tensor = tensor.view(1, 1, -1)
        # if tensor.size()[2] < 3:
        #     tensor = tensor.expand(tensor.size()[0], tensor.size()[1], 3)

        # Pad to fit width
        assert tensor.numel() % 3 == 0, 'Tensor should be RGB now'
        missing_values = self._compute_padding(math.ceil(tensor.numel() / 3), width)
        tensor = torch.cat([tensor.view([-1, 3]), pad_color.expand([missing_values, 3])])

        column_tensor = tensor.view([-1, width, 3])
        count = math.ceil(column_tensor.size()[0] / height)

        # result_dims = [
        #     math.ceil(count / self._items_per_row) * height,
        #     self._items_per_row * width,
        #     3
        # ]
        # Pad column tensor so it can be viewed as canvas of result_dims
        column_height = column_tensor.size()[0]
        missing_image_rows = self._compute_padding(column_height, height)
        missing_images = self._compute_padding(count, self._items_per_row)

        pad_tensor = pad_color.expand((missing_image_rows + missing_images * height, width, 3))

        padded_column_tensor = torch.cat([column_tensor, pad_tensor]).view([-1, width, 3])

        # Compute tiling indices
        image_rows = padded_column_tensor.size(0)
        indices = self._column_tiling_indices(str(column_tensor.device), image_rows, self._items_per_row, height)

        # Reorder tensor in order to make tiling
        images = padded_column_tensor.index_select(0, indices)
        return images.view([-1, self._items_per_row * width, 3]), TensorViewProjectionUIParams(width, height,
                                                                                               self._items_per_row,
                                                                                               count)

    @staticmethod
    def _squash_all_dims_but_last(original_dims: List[int]) -> List[int]:
        """Collect all the dimensions but the last one.

        Intention: provide a 2D interpretation of the ND tensor for UI.
        """
        product = 1
        for dim in original_dims:
            product *= dim

        result = int(product / original_dims[-1])
        return [result, original_dims[-1]]

    @staticmethod
    def _colorize(data: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
        """Colorize data.

        Interval (-inf, -maximum) is clipped to value 1 - red color
        Interval (-maximum, -minimum) is scaled linearly to (1, 0) - red color
        Interval (-minimum, minimum) is clipped to value 0 -  black color
        Interval (minimum, maximum) is scaled linearly to (0, 1) - green color
        Interval (maximum, +inf) is clipped to value 1 - green color
        Value -inf is set to value 1 - magenta
        Value +inf is set to value 1 - cyan
        Value NaN is set to value 1 - blue
        """
        data = data.float()

        # print(f'Device {data.device}')
        # define colors
        negative_color = torch.tensor([1.0, 0.0, 0.0], device=data.device)
        positive_color = torch.tensor([0.0, 1.0, 0.0], device=data.device)
        nan_color = torch.tensor([0.0, 0.0, 1.0], device=data.device)
        positive_inf_color = torch.tensor([0.0, 1.0, 1.0], device=data.device)
        negative_inf_color = torch.tensor([1.0, 0.0, 1.0], device=data.device)

        # prepare substitution masks
        mask_positive = data > minimum
        mask_negative = data < -minimum

        mask_positive_clip = data >= maximum
        mask_negative_clip = data <= -maximum
        mask_nan = torch.isnan(data)
        inf = float('inf')
        ninf = -float('inf')
        mask_positive_inf = data == inf
        mask_negative_inf = data == ninf

        # create result
        result_dims = data.size() + (3,)
        result = torch.zeros(result_dims, device=data.device)

        # linear scaling of negative values
        if mask_negative.any():
            zeros = torch.zeros(result_dims, device=data.device)
            zeros[mask_negative] = negative_color
            processed_data = (-data - minimum) / (maximum - minimum)
            result += zeros * processed_data.unsqueeze(data.dim())

        # linear scaling of positive values
        if mask_positive.any():
            zeros = torch.zeros(result_dims, device=data.device)
            zeros[mask_positive] = positive_color
            processed_data = (data - minimum) / (maximum - minimum)
            result += zeros * processed_data.unsqueeze(data.dim())

        # substitute fixed values
        color_substitutions = [
            (mask_positive_clip, positive_color),
            (mask_negative_clip, negative_color),
            (mask_nan, nan_color),
            (mask_positive_inf, positive_inf_color),
            (mask_negative_inf, negative_inf_color),
        ]
        for mask, color in color_substitutions:
            if mask.any():
                result[mask] = color

        return result

    @staticmethod
    def _rgb_transform(data: torch.Tensor, minimum: float, maximum: float):
        data = data.float()
        data = (data - minimum) / (maximum - minimum)

        # prepare substitution masks
        mask_max_clip = data > 1.0
        mask_min_clip = data < 0.0

        # substitute fixed values
        color_substitutions = [
            (mask_max_clip, 1.0),
            (mask_min_clip, 0.0),
        ]
        for mask, color in color_substitutions:
            if mask.any():
                data[mask] = color

        return data
    
    def _inverse_transform_coordinates(self, dims: List[int], x: int, y: int) -> int:
        height, width = self._compute_tile_dimensions(dims, False)
        tile_x = math.floor(x / width)
        tile_y = math.floor(y / height)
        tile_index = tile_y * self._items_per_row + tile_x
        pos_in_tile_x = x % width
        pos_in_tile_y = y % height

        row = tile_index * height + pos_in_tile_y
        column = pos_in_tile_x
        position = row * width + column
        return position

    def value_at(self, tensor: torch.Tensor, x: int, y: int) -> float:
        # TODO: We perform the transform twice per observation phase - make this more neat
        tensor = self.apply_any_transforms(tensor)

        position = self._inverse_transform_coordinates(list(tensor.size()), x, y)
        if position >= tensor.numel():
            return float('nan')
        else:
            return tensor.view([-1])[position].item()


def update_scale_to_respect_minimum_size(tensor: torch.Tensor, minimal_size: int, scale_set_by_user: int):
    height, width = tensor.size()[-3], tensor.size()[-2]
    scale_factor = minimal_size / min(height, width)
    if scale_factor > 1 and scale_factor > scale_set_by_user:
        return math.ceil(scale_factor)
    else:
        return scale_set_by_user


class TensorObservable(ABC, MemoryBlockObservable):
    _scale: int = 4
    _scale_set_by_user: int = 1
    _is_rgb: bool = False
    _tensor_view_projection: TensorViewProjection
    # minimum observer size in pixels, used for automatic rescaling of observers which are too small
    _tensor: torch.Tensor = None

    def __init__(self):
        super().__init__()
        self._tensor_view_projection = TensorViewProjection(is_buffer=False)

    @abstractmethod
    def get_tensor(self) -> Optional[torch.Tensor]:
        pass

    def _update_scale_to_respect_minimum_size(self, tensor):
        self._scale = update_scale_to_respect_minimum_size(tensor,
                                                           GlobalSettings.instance().observer_memory_block_minimal_size,
                                                           self._scale_set_by_user)

    def get_data(self) -> TensorObservableData:
        self._tensor = self.get_tensor()
        if self._tensor is None:
            return dummy_tensor_observable_data()

        self._tensor = sanitize_tensor(self._tensor)

        if not self._tensor.is_contiguous():
            # TODO: show this warning only once
            # logger.warning(f"Tensor {self._tensor.size()} is not contiguous.")
            self._tensor = self._tensor.contiguous()

        tensor, projection_params = self._tensor_view_projection.transform_tensor(self._tensor, self._is_rgb)

        self._update_scale_to_respect_minimum_size(tensor)

        # Do not scale now - it's done in frontend using params.scale
        # data = self._scale_tensor(data, self._scale)
        params = TensorObservableParams(
            scale=self._scale,
            projection=projection_params
        )
        return TensorObservableData(tensor, params)

    def get_properties(self) -> List[ObserverPropertiesItem]:
        def update_scale(value: str):
            self._scale_set_by_user = int(value)
            return value

        def update_is_rgb(value: str):
            self._is_rgb = bool(value)
            return value

        return [
                   ObserverPropertiesItem('Scale', 'number', self._scale_set_by_user, update_scale),
                   ObserverPropertiesItem('RGB', 'checkbox', self._is_rgb, update_is_rgb),
               ] + self._tensor_view_projection.get_properties()

    @staticmethod
    def _scale_tensor(data: torch.Tensor, scale: int) -> torch.Tensor:
        def get_indices(size: int):
            return [v for x in range(size) for v in [x] * scale]

        x = get_indices(data.size()[0])
        y = get_indices(data.size()[1])
        return data[:, y][x]

    def request_callback(self, data: RequestData):
        x = int(data.data['x'])
        y = int(data.data['y'])
        if self._tensor is None or self._is_rgb:
            value = float('nan')
        else:
            value = self._tensor_view_projection.value_at(self._tensor, x, y)

        return {
            "value": 'NaN' if math.isnan(value) else value
        }

    def get_callbacks(self) -> ObserverCallbacks:
        return ObserverCallbacks().add_request(self.request_callback)
