import math
from typing import Tuple

import torch
import numpy as np
import logging

from torchsim.core import FLOAT_NAN, get_float
from torchsim.core.utils.tensor_utils import id_to_one_hot

logger = logging.getLogger(__name__)


class SpaceDivisor:
    """Convert 2D space positions into unique landmarks.

    Divide the square space (coordinates from 0 to MAX_POSITION) into
    horizontal_segments*vertical_segments of unique landmarks. For each given position return the nearest landmark.
    """

    #  TODO (Feat): add the support for non-square maps
    MAX_POSITION = 1.0

    horizontal_segments: int
    vertical_segments: int
    _device: str
    num_landmarks: int
    one_hot: bool

    def __init__(self, horizontal_segments, vertical_segments, device):
        """Gets a tensor of YX positions and for each finds the corresponding landmark (ID of a area in the map).

        Args:
            horizontal_segments: how many horizontal landmarks we want
            vertical_segments: how many vertical landmarks we want
        """
        self._device = device
        self._float_dtype = get_float(device)
        self.vertical_segments = vertical_segments
        self.horizontal_segments = horizontal_segments
        self.num_landmarks = self.vertical_segments * self.horizontal_segments

        # compute size of one segment (for both x and y)
        self.segment_sizes = self.MAX_POSITION /\
                             torch.tensor([self.vertical_segments, self.horizontal_segments], dtype=self._float_dtype, device=self._device)

        self.nans = torch.tensor((self.num_landmarks,), device=self._device, dtype=self._float_dtype)
        self.nans.fill_(FLOAT_NAN)

    def _handle_nan(self):
        result = torch.tensor([FLOAT_NAN], dtype=self._float_dtype, device=self._device)
        return result, self.nans

    def get_landmark_normalize(self, y: float, x: float, minimum: float, maximum: float):
        if math.isnan(y) or math.isnan(x):
            return self._handle_nan()
        _range = maximum - minimum
        x = max((x - minimum) / _range, 0)
        y = max((y - minimum) / _range, 0)
        return self.get_landmark(y, x)

    def get_landmark(self, y: float, x: float):
        """Finds a nearest landmark for a given yx position. Position expected from range [0, MAX_POSITION).

        Args:
            y: y position in range [0,MAX_POSITION)
            x: x position in range [0,MAX_POSITION)

        Returns:
            ID of the nearest landmark from [0, horizontal_segments * vertical_segments] as a scalar Tensor
            (or one-hot).
        """
        if math.isnan(y) or math.isnan(x):
            return self._handle_nan()

        sy = self.segment_sizes[0].item()
        sx = self.segment_sizes[1].item()

        y_landmark = np.floor(y / sy)
        x_landmark = np.floor(x / sx)

        res = y_landmark * self.horizontal_segments + x_landmark

        result = torch.tensor([res], dtype=self._float_dtype, device=self._device)
        return result, id_to_one_hot(result.long(), self.num_landmarks).squeeze(0)

    def get_landmarks(self, yx_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute landmarks for no_positions stored in a 2D tensor.

        Args:
            yx_positions: tensor of size [num_positions, 2] containing list of YX positions in the square map

        Returns:
            Tensor of size [num_positions] (or [num_positions, num_landmarks]), which contains nearest landmark
            for each position.
        """
        yx_landmarks = torch.floor(yx_positions / self.segment_sizes)
        result = (yx_landmarks[:, 0] * self.horizontal_segments) + yx_landmarks[:, 1]

        one_hot_result = id_to_one_hot(result.view(-1).long(), self.num_landmarks)

        return result, one_hot_result


