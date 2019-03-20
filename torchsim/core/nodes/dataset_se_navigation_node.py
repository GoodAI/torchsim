import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

import torch
from torchsim.core import FLOAT_NEG_INF, FLOAT_NAN
from torchsim.core.datasets.dataset_se_base import SeDatasetSize, DatasetSeBase
from torchsim.core.datasets.dataset_se_task_one import DatasetSeTask1
from torchsim.core.datasets.space_divisor import SpaceDivisor
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorOutputs, SpaceEngineersConnectorInputs
from torchsim.gui.observables import disable_on_runtime
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.validators import *
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed

logger = logging.getLogger(__name__)


class SamplingMethod(Enum):
    ORDERED = 0,         # in the order it was created - correlated in time, movement was continuous
    RANDOM_ORDER = 1,    # randomly permuted images
    RANDOM_SAMPLING = 2  # randomly sampled with replacement


@dataclass
class DatasetSENavigationParams(ParamsBase):
    """Class used for configuring the world."""
    dataset_size: SeDatasetSize = SeDatasetSize.SIZE_24  # size of the dataset to be loaded
    sampling_method: SamplingMethod = SamplingMethod.ORDERED  # in what order to present the samples
    horizontal_segments: int = 3  # num of horizontal segments for discretized 2D position into landmark
    vertical_segments: int = 3  # num of vertical segments for discretized 2D position into landmark
    channel_first: bool = False  # if true, the format is [channel,Y,X], else [Y,X,channel] (default)

    @property
    def dataset_dims(self) -> Tuple[int, int]:
        return self.dataset_size.value, self.dataset_size.value


class DatasetSeNavigationUnit(Unit):
    """Loads the dataset with images from SE and their corresponding positions.

    Then presents each image and position either sequentially or in a random order.
    """

    _params: DatasetSENavigationParams

    _images: torch.Tensor  # whole data loaded from the file
    _positions: torch.Tensor  # whole positions loaded from the file
    _landmarks: torch.Tensor  # position point converted into a single landmark value

    last_image: torch.Tensor
    last_position: torch.Tensor
    last_landmark: torch.Tensor
    last_landmark_one_hot: torch.Tensor

    task_id_const: torch.Tensor
    unused_output: torch.Tensor
    testing_phase_indicator_const: torch.Tensor

    _random: np.random.RandomState

    _n_samples: int

    _pos: int

    def __init__(self, creator: TensorCreator, params: DatasetSENavigationParams, random: np.random.RandomState):
        super().__init__(creator.device)
        self._params = params.clone()
        self._random = random

        dataset = DatasetSeTask1(self._params.dataset_size)
        size, images, labels = dataset.get_all()
        self._img_size = size
        self._n_samples = len(images)
        self._positions_permuted = None

        # load the images and positions
        self._images = images.type(self._float_dtype).to(self._device)

        self._positions = labels.type(self._float_dtype).to(self._device) / 100.0

        # compute landmarks for each position
        divisor = SpaceDivisor(self._params.horizontal_segments,
                               self._params.vertical_segments,
                               self._device)
        self._landmarks, self._landmarks_one_hot = divisor.get_landmarks(self._positions)

        # prepare the tensors for the first step
        if self._params.channel_first:
            self._images = self._images.permute(0, 3, 1, 2)
            img_size = [DatasetSeBase.N_CHANNELS, *self._params.dataset_dims]
        else:
            img_size = [*self._params.dataset_dims, DatasetSeBase.N_CHANNELS]

        self.last_image = creator.full(img_size,
                                       fill_value=FLOAT_NAN,
                                       dtype=self._float_dtype,
                                       device=self._device)

        self.last_position = creator.full(self._positions[0].shape,
                                          fill_value=FLOAT_NAN,
                                          dtype=self._float_dtype,
                                          device=self._device)

        self.last_landmark = creator.full(self._landmarks[0].shape,
                                          fill_value=FLOAT_NAN,
                                          dtype=self._float_dtype,
                                          device=self._device)
        self.last_landmark_one_hot = creator.full(self._landmarks_one_hot[0].shape,
                                                  fill_value=FLOAT_NAN,
                                                  dtype=self._float_dtype,
                                                  device=self._device)

        self.task_id_const = creator.full([1], 1.0, dtype=self._float_dtype, device=self._device)
        self.testing_phase_indicator_const = creator.full([1], 0.0, dtype=self._float_dtype, device=self._device)
        self.unused_output = creator.full([1], FLOAT_NEG_INF, dtype=self._float_dtype, device=self._device)

        self._pos = -1
        self._pos_permuted = -1

    def step(self):
        """Each step reads a new image and position (either sequentially or randomly)."""

        if self._params.sampling_method == SamplingMethod.RANDOM_SAMPLING:
            self._pos = self._random.randint(low=0, high=self._n_samples)
        elif self._params.sampling_method == SamplingMethod.RANDOM_ORDER:
            if self._positions_permuted is None:
                self._positions_permuted = self._random.permutation(range(self._n_samples))
            self._pos_permuted = (self._pos_permuted + 1) % self._n_samples
            self._pos = self._positions_permuted[self._pos_permuted]
        elif self._params.sampling_method == SamplingMethod.ORDERED:
            self._pos = (self._pos + 1) % self._n_samples
        else:
            raise NotImplementedError(f"Sampling behavior not implemented for type {self._params.sampling_method}")

        # read the next sample
        self.last_image.copy_(self._images[self._pos].type(self._float_dtype) / 255.0)
        self.last_position.copy_(self._positions[self._pos])
        self.last_landmark.copy_(self._landmarks[self._pos])
        self.last_landmark_one_hot.copy_(self._landmarks_one_hot[self._pos])


class DatasetSeNavigationOutputs(SpaceEngineersConnectorOutputs):
    """All outputs of the DatasetSeNavigationNode.

    image_output: width by height RGB image.

    task_to_agent_location: [X,Y] position of the character in SE Task1.
    task_to_agent_location_int: single value landmark position of the character in SE Task1.
    task_to_agent_location_one_hot: one-hot representation of tha landmark value.

    metadata_task_id: information about the task id.
    metadata_testing_phase: information if the sample is from the training/testing phase of the Task1.

    reward_output: blank output.
    task_to_agent_label: blank output.
    task_to_agent_location_target: blank output.
    task_to_agent_target: blank output.
    task_to_agent_location_target: blank output.
    metadata_task_instance_id: blank output.
    metadata_task_status: blank output.
    metadata_task_instance_status: blank output.
    """

    def __init__(self, owner):
        super().__init__(owner)

    def prepare_slots(self, unit: DatasetSeNavigationUnit):
        self.image_output.tensor = unit.last_image
        self.task_to_agent_location.tensor = unit.last_position
        self.task_to_agent_location_int.tensor = unit.last_landmark
        self.task_to_agent_location_one_hot.tensor = unit.last_landmark_one_hot

        self.metadata_task_id.tensor = unit.task_id_const
        self.metadata_testing_phase.tensor = unit.testing_phase_indicator_const

        self.reward_output.tensor = unit.unused_output
        self.task_to_agent_label.tensor = unit.unused_output
        self.task_to_agent_location_target.tensor = unit.unused_output
        self.task_to_agent_target.tensor = unit.unused_output
        self.task_to_agent_location_target_one_hot.tensor = unit.unused_output
        self.metadata_task_instance_id.tensor = unit.unused_output
        self.metadata_task_status.tensor = unit.unused_output
        self.metadata_task_instance_status.tensor = unit.unused_output


class DatasetSeNavigationNode(WorkerNodeBase[SpaceEngineersConnectorInputs, DatasetSeNavigationOutputs]):
    """Generates a samples for Task1 from an external dataset.

    Parameters of the samples generation are maintained by DatasetSENavigationParams class.
    In case of random samples generation it is possible to set the seed value.
    """

    _unit: DatasetSeNavigationUnit
    _seed: int
    outputs: DatasetSeNavigationOutputs

    def __init__(self, params: DatasetSENavigationParams, seed: int = None, name="DatasetSENavigation"):
        super().__init__(name=name, outputs=DatasetSeNavigationOutputs(self),
                         inputs=SpaceEngineersConnectorInputs(self))
        self._params = params.clone()
        self._seed = seed

    def _create_unit(self, creator: TensorCreator):
        random_generator = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)

        return DatasetSeNavigationUnit(creator, self._params, random_generator)

    @property
    def dataset_size(self) -> SeDatasetSize:
        return self._params.dataset_size

    @dataset_size.setter
    def dataset_size(self, value: SeDatasetSize):
        self._params.dataset_size = value

    @property
    def sampling_method(self) -> SamplingMethod:
        return self._params.sampling_method

    @sampling_method.setter
    def sampling_method(self, value: SamplingMethod):
        self._params.sampling_method = value

    @property
    def horizontal_segments(self) -> int:
        return self._params.horizontal_segments

    @horizontal_segments.setter
    def horizontal_segments(self, value: int):
        validate_positive_int(value)
        self._params.horizontal_segments = value

    @property
    def vertical_segments(self) -> int:
        return self._params.vertical_segments

    @vertical_segments.setter
    def vertical_segments(self, value: int):
        validate_positive_int(value)
        self._params.vertical_segments = value

    @property
    def channel_first(self) -> bool:
        return self._params.channel_first

    @channel_first.setter
    def channel_first(self, value: bool):
        self._params.channel_first = value

    @property
    def dataset_dims(self) -> List[int]:
        return [self.dataset_size.value, self.dataset_size.value]

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        validate_positive_int(value)
        self._seed = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('Dataset size', type(self).dataset_size, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Sampling method', type(self).sampling_method, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Horizontal segments', type(self).horizontal_segments, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Vertical segments', type(self).vertical_segments, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Channel first', type(self).channel_first, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Dataset dimmensions', type(self).dataset_dims, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Seed', type(self).seed, edit_strategy=disable_on_runtime)
        ]

    def _step(self):
        self._unit.step()
