import math
from dataclasses import dataclass
from typing import Tuple

import matplotlib.cm as cm
import matplotlib.colors as colors
import torch.nn.functional as F

import torch
from torchsim.core.graph.node_base import EmptyOutputs
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeWithInternalsBase
from torchsim.core.memory.tensor_creator import TensorCreator
# TODO: Discounting by 0.99 in time
# TODO: parameterize colormap, scales
# TODO: rescale dynamically
# TODO: abstract params class
# TODO: create abstract node that has params, parse_vars, etc
from torchsim.core.models.expert_params import ParamsBase
from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime
from torchsim.gui.validators import *


@dataclass
class CartographicParams(ParamsBase):
    shape: Tuple[int, int, int] = (64, 64, 3)  # size of display area
    base_shape: Tuple[int, int] = (10, 10)  # size of original SE grid area
    kernel_shape: Tuple[int, int] = (3, 3)  # shape of smoothing kernel
    base_multiplier: float = 0
    colormap: str = 'Blues'  # colormap used for visualisation (see https://goo.gl/sjYGt4)
    cmin: int = 0  # minimal value for colormap
    cmax: int = 1  # maximal value for colormap
    smoothing: bool = False  # apply smoothing to map
    draw_target: bool = True  # toggle to draw target location


class CartographicNodeUnit(Unit):
    _params: CartographicParams
    
    def __init__(self, creator: TensorCreator, device, params: CartographicParams):

        self._params = params.clone()

        super().__init__(device)

        self.map_rgb = creator.zeros(
            self._params.shape, device=device)
        self.map_raw = creator.zeros(
            self._params.shape[:-1], device=device)

        self.cmap = cm.get_cmap(
            self._params.colormap)
        self.cnorm = colors.Normalize(
            vmin=self._params.cmin,
            vmax=self._params.cmax)

        self.initdone = False
        self.tartget_x = 0
        self.tartget_y = 0

        # Smoothing Kernel
        if self._params.smoothing:
            self.kernel = (torch.ones(
                (self._params.kernel_shape[0],
                 self._params.kernel_shape[1]))) / (self._params.kernel_shape[0] * self._params.kernel_shape[1])
            self.kernel = self.kernel.to(device='cuda')
            self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def smooth(raw_map, kernel):
        """Smooth a 2D/3D rgb map using a convolutional kernel.

        Args:
            raw_map: 2D map of values to be smoothed
            kernel: convolutional kernel for smoothing

        Returns:
            Smoothed map.
        """
        if len(raw_map.shape) == 2:
            raw_map = raw_map.unsqueeze(0).unsqueeze(0)
            raw_map = F.conv2d(raw_map, kernel, padding=1)
            return raw_map.squeeze()
        else:
            raw_map = raw_map.permute((2, 0, 1))
            raw_map = raw_map.unsqueeze(0)
            raw_map[:, 0, :, :] = F.conv2d(
                raw_map[:, 0, :, :].unsqueeze(0),
                kernel, padding=1)
            raw_map[:, 1, :, :] = F.conv2d(
                raw_map[:, 1, :, :].unsqueeze(0),
                kernel, padding=1)
            raw_map[:, 2, :, :] = F.conv2d(
                raw_map[:, 2, :, :].unsqueeze(0),
                kernel, padding=1)
            raw_map = raw_map.squeeze()
            return raw_map.permute((1, 2, 0))

    def minmax(self, value):
        """Store running min/max value."""
        self.min = value if value < self.min else self.min
        self.max = value if value > self.max else self.max

    def renormalize(self) -> colors.Normalize:
        """Set normalization bounds for colormap."""
        return colors.Normalize(
            vmin=self.min,
            vmax=self.max)

    def map_to_base_coords(
            self, x: float, y: float) -> Tuple[int, int]:
        """Converts SE coordinates to visualisation map coordinates.

        Args:
            x: x-location from SE
            y: y-location from SE

        Returns:
            New coordinates within the map.
        """
        if math.isnan(x) or math.isnan(y):
            return 0, 0
        else:
            x_base = math.floor(x * self._params.base_multiplier)
            y_base = self._params.shape[1] - 1 - math.floor(y * self._params.base_multiplier)
            return x_base, y_base

    def to_rgb(self, value_map):
        """Converts value map to rgb tensor.

        Args:
            value_map: map of original values
        """
        self.map_rgb[:, :, :] = torch.from_numpy(
            (self.cmap(self.cnorm(value_map)))[:, :, :-1])

    def step(self, obs, location, location_target):
        """Simulation step that collects observations and plots them onto the cartographic map.

        Args:
            obs: 1D value to be assigned to current location
            location: Current location of agent
            location_target: Where the agent should go
        """

        # init step - map target
        if not self.initdone:
            self.to_rgb(self.map_raw)
            self.initdone = True

        # value to be visualised
        value = obs.tensor.item()

        # perform mapping between original coordinates and
        # display coordinates
        x, y = self.map_to_base_coords(
            location.tensor[0].item(),
            location.tensor[1].item())

        self.map_raw[y, x] = value

        if self._params.smoothing:
            self.to_rgb(self.smooth(self.map_raw, self.kernel))
        else:
            self.to_rgb(self.map_raw)

        if self._params.draw_target:
            self.tartget_x, self.tartget_y = self.map_to_base_coords(
                location_target.tensor[0].item(),
                location_target.tensor[1].item())
            self.map_rgb[self.tartget_y, self.tartget_x, 0] = 250
            self.map_rgb[self.tartget_y, self.tartget_x, 1] = 0
            self.map_rgb[self.tartget_y, self.tartget_x, 2] = 0


class CartographicNodeInputs(Inputs):
    """Node INPUTS."""

    def __init__(self, owner):
        super().__init__(owner)
        self.observation = self.create(
            "observation")
        self.task_location = self.create(
            "task_location_output")
        self.task_location_target = self.create(
            "task_location_target_output")


class CartographicNodeInternals(MemoryBlocks):
    """Node INTERNALS."""

    def __init__(self, owner):
        super().__init__(owner)
        self.map_raw = self.create("output_raw")
        self.map_rgb = self.create("output_rgb")

    def prepare_slots(self, unit: CartographicNodeUnit):
        self.map_raw.tensor = unit.map_raw
        self.map_rgb.tensor = unit.map_rgb


class CartographicNode(WorkerNodeWithInternalsBase[CartographicNodeInputs, CartographicNodeInternals, EmptyOutputs]):
    """A node that allows for cartographic visualisation of agent's perception of the world in which it operates."""
    inputs: CartographicNodeInputs

    def __init__(self, device="cuda", name="Value"):

        super().__init__(
            name=f"Cartographic Map::{name}",
            inputs=CartographicNodeInputs(self),
            memory_blocks=CartographicNodeInternals(self))

        self._device = device
        self._params = CartographicParams()
        self._params.base_multiplier = self._params.shape[0] / self._params.base_shape[0]

        assert len(self._params.shape) == 3, 'shape must be 3D (RGB)'
        assert self._params.shape[0] == self._params.shape[1], 'Only square maps are allowed'

    def _create_unit(self, creator: TensorCreator):
        return CartographicNodeUnit(
            creator, self._device, self._params)

    @property
    def shape(self) -> List[int]:
        return list(self._params.shape)

    @shape.setter
    def shape(self, value: List[int]):
        self._params.shape = tuple(value)

    @property
    def base_shape(self) -> List[int]:
        return list(self._params.base_shape)

    @base_shape.setter
    def base_shape(self, value: List[int]):
        self._params.base_shape = tuple(value)

    @property
    def kernel_shape(self) -> List[int]:
        return list(self._params.kernel_shape)

    @kernel_shape.setter
    def kernel_shape(self, value: List[int]):
        self._params.kernel_shape = tuple(value)

    @property
    def base_multiplier(self) -> float:
        return self._params.base_multiplier

    @base_multiplier.setter
    def base_multiplier(self, value: float):
        validate_positive_float(value)
        self._params.base_multiplier = value

    @property
    def colormap(self) -> str:
        return self._params.colormap

    @colormap.setter
    def colormap(self, value: str):
        self._params.colormap = value

    @property
    def cmin(self) -> int:
        return self._params.cmin

    @cmin.setter
    def cmin(self, value: int):
        validate_positive_with_zero_int(value)
        self._params.cmin = value

    @property
    def cmax(self) -> int:
        return self._params.cmax

    @cmax.setter
    def cmax(self, value: int):
        validate_positive_with_zero_int(value)
        self._params.cmax = value

    @property
    def smoothing(self) -> bool:
        return self._params.smoothing

    @smoothing.setter
    def smoothing(self, value: bool):
        self._params.smoothing = value

    @property
    def draw_target(self) -> bool:
        return self._params.draw_target

    @draw_target.setter
    def draw_target(self, value: bool):
        self._params.draw_target = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Defines which parameters are accessible from GUI."""
        return [
            self._prop_builder.auto('Shape', type(self).shape, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Base shape', type(self).base_shape, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Kernel shape', type(self).kernel_shape, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Base multiplier', type(self).base_multiplier, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Colormap', type(self).colormap, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('C_min', type(self).cmin, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('C_max', type(self).cmax, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Smoothing', type(self).smoothing, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Draw target', type(self).draw_target, edit_strategy=disable_on_runtime)
        ]

    def _step(self):
        self._unit.step(
            self.inputs.observation,
            self.inputs.task_location,
            self.inputs.task_location_target)
