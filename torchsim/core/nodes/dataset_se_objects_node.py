import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

import numpy as np

import torch
from torchsim.core import FLOAT_NAN, FLOAT_NEG_INF
from torchsim.core.datasets.dataset_se_base import SeDatasetSize, DatasetSeBase
from torchsim.core.datasets.dataset_se_task_zero import DatasetSeTask0
from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeWithInternalsBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorOutputs, SpaceEngineersConnectorInputs
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.validators import *
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed

logger = logging.getLogger(__name__)


class DatasetConfig(Enum):
    TRAIN_ONLY = 0
    TEST_ONLY = 1
    TRAIN_TEST = 3


@dataclass
class DatasetSeObjectsParams(ParamsBase):
    """Class used for configuring the world."""

    filename_with_path: str = None  # path to the pkl file to be loaded
    seed: int = 1  # deterministic
    class_filter: Optional[List[int]] = None
    random_order: bool = False  # present the samples in random order
    save_gpu_memory: bool = False
    location_filter_ratio: float = 1.0  # filter objects by their locations around the center (1: no filter)
    dataset_size: SeDatasetSize = SeDatasetSize.SIZE_24  # smallest resolution by default
    dataset_config: DatasetConfig = DatasetConfig.TRAIN_ONLY
    switch_train_resets_train_pos: bool = True  # True: reset the training_pos to 0
    is_hide_labels: bool = False  # False: do not hide labels from the clients

    @property
    def dataset_dims(self) -> Tuple[int, int]:
        return self.dataset_size, self.dataset_size


class DatasetSeObjectsUnit(Unit):
    """Loads the dataset with images from SE and their corresponding labels.

    Then presents each image and label either sequentially or in a random order.
    """
    NUM_LABELS = 20

    _params: DatasetSeObjectsParams

    _train_images: torch.Tensor  # train data loaded from the file
    _train_labels: torch.Tensor  # test data loaded from the file
    _train_instance_ids: torch.Tensor  # train labels loaded from the file
    _train_examples_per_class: torch.Tensor  # train labels loaded from the file

    _test_images: torch.Tensor  # train labels loaded from the file
    _test_labels: torch.Tensor  # test labels loaded from the file
    _test_instance_ids: torch.Tensor  # train labels loaded from the file
    _test_examples_per_class: torch.Tensor  # train labels loaded from the file

    last_image: torch.Tensor
    last_label: torch.Tensor
    last_truth: torch.Tensor
    last_instance_id: torch.Tensor
    last_examples_per_class: torch.Tensor

    unused_output: torch.Tensor
    task_id_const: torch.Tensor
    testing_phase_indicator: torch.Tensor

    training_pos: torch.Tensor  # position in the training set - for the serialization

    _random: np.random

    _pos: int

    _train_label_indexes: List[int]  # indexes of labels in filtered training data
    _test_label_indexes: List[int]  # indexes of labels in filtered testing data

    _num_classes: int
    _is_training: bool
    _presented: int

    _labels_hidden: bool  # simulates testing, but on the training data (just hide the labels and continue)

    _skip_next_step: int  # how many frames to skip on the next step

    def __init__(self, creator: TensorCreator, params: DatasetSeObjectsParams, random):
        super().__init__(creator.device)
        self._params = params
        self._random = random
        self._save_memory = params.save_gpu_memory

        dataset = DatasetSeTask0(self._params.dataset_size)
        size, train, test = dataset.get_all()

        self._train_instance_ids = train[2].to('cpu')
        self._test_instance_ids = test[2].to('cpu')

        self._img_size = size

        # load the images and labels
        if self._save_memory:
            device = 'cpu'
        else:
            device = self._device
        self._train_images = train[0].to(device)
        self._train_labels = train[1].to(self._device)
        self._test_images = test[0].to(device)
        self._test_labels = test[1].to(self._device)

        # prepare the tensors for the first step
        img_size = [*self._params.dataset_dims, DatasetSeBase.N_CHANNELS]
        num_classes = self._train_labels.shape[1]

        self.last_image = self._create_tensor(img_size, FLOAT_NAN, creator)
        self.last_label = self._create_tensor([num_classes], FLOAT_NAN, creator)
        self.last_truth = self._create_tensor([num_classes], FLOAT_NAN, creator)
        self.hidden_label = self._create_tensor([num_classes], FLOAT_NAN, creator)
        self.last_instance_id = self._create_tensor([1], FLOAT_NAN, creator)

        self.task_id_const = self._create_tensor([1], 0.0, creator)
        self.unused_output = self._create_tensor([1], FLOAT_NEG_INF, creator)
        self.testing_phase_indicator = self._create_tensor([1], FLOAT_NAN, creator)

        self.training_pos = creator.full([1], fill_value=-1, dtype=torch.long, device=self._device)

        self._reset_indexes_and_filtering()

    def _reset_indexes_and_filtering(self):
        # apply the class_filter if necessary (time consuming)
        self._filter_labels()
        self._pos = -1

        self._presented = 0

        # TODO might be broken with the new labels_hidden set from GUI
        self._labels_hidden = False
        self._is_training = (self._params.dataset_config == DatasetConfig.TRAIN_ONLY or
                             self._params.dataset_config == DatasetConfig.TRAIN_TEST)
        self._skip_next_step = 0

        # write resetted value to the tensor if possible
        try:
            self.training_pos[0] = self._pos
        except Any:
            logger.warning("could not write to tensor (initialization?)")

    def _create_tensor(self, shape: List[int], fill_value: float, creator: TensorCreator):
        return creator.full(shape, fill_value=fill_value, dtype=self._float_dtype, device=self._device)

    def step(self):
        """Each step read label and image (either sequentially or randomly)."""

        if self._is_training:
            self._pos = self.training_pos.item()

        if self._params.dataset_config == DatasetConfig.TRAIN_ONLY:
            sample_range = len(self._train_label_indexes)
            self._is_training = True
        elif self._params.dataset_config == DatasetConfig.TEST_ONLY:
            sample_range = len(self._test_label_indexes)
            self._is_training = False
        else:
            total_dataset_size = len(self._train_label_indexes) + len(self._test_label_indexes)
            self._is_training = self._presented % total_dataset_size < len(self._train_label_indexes)
            switch_from_training_to_testing = self._presented % total_dataset_size == len(self._train_label_indexes)
            switch_from_testing_to_training = self._presented % total_dataset_size == 0
            if switch_from_training_to_testing or switch_from_testing_to_training:
                self._pos = -1  # needed for sequential order
            if self._is_training:
                sample_range = len(self._train_label_indexes)
            else:
                sample_range = len(self._test_label_indexes)

        self._presented += 1

        if self._params.random_order:
            self._pos = self._random.randint(low=0, high=sample_range)
            if self._is_location_filtering():
                self._pos = self._filter_location_random_position(self._is_training)
        else:  # sequential order
            self._pos = (self._pos + 1) % sample_range
            if self._is_location_filtering():
                skipped_beginning, self._skip_next_step, self._pos = self._filter_location_sequential(self._is_training)
                self._presented += skipped_beginning

        self._copy_to_outputs_from(self._pos, self._is_training)

        self._pos += self._skip_next_step
        self._presented += self._skip_next_step
        self._skip_next_step = 0

        # write the training pos to tensor
        if self._is_training:
            self.training_pos[0] = self._pos

    def _filter_location_sequential(self, training: bool):
        skipped_beginning = 0
        skipped_end = 0
        position = self._pos  # index

        pos_min, pos_max, inst_min, inst_max = self.get_instance_location_range(training, position)

        if position < pos_min:
            skipped_beginning = pos_min - position
            position = pos_min
        if position == pos_max:
            skipped_end = inst_max - position
        assert position <= pos_max, "caller makes sure position never exceeds pos_max"

        return skipped_beginning, skipped_end, position

    def _filter_location_random_position(self, training: bool):

        position = self._pos
        pos_min, pos_max, inst_min, inst_max = self.get_instance_location_range(training, position)

        position = self._random.randint(low=pos_min, high=pos_max + 1)
        return position

    def get_instance_location_range(self, training: bool, pos: int):

        if training:
            instance_indexes = self._train_label_indexes
            instance_ids = self._train_instance_ids
        else:
            instance_indexes = self._test_label_indexes
            instance_ids = self._test_label_indexes

        instance_id = instance_ids[instance_indexes[pos]]
        inst_min = pos
        inst_max = pos
        while inst_min > 0 and instance_ids[instance_indexes[inst_min - 1]] == instance_id:
            inst_min -= 1
        while len(instance_indexes) > inst_max + 1 and instance_ids[instance_indexes[inst_max + 1]] == instance_id:
            inst_max += 1
        inst_range = inst_max - inst_min + 1

        inst_allowed_range = int(self._params.location_filter_ratio * inst_range)
        inst_allowed_range = max(1, inst_allowed_range)
        inst_allowed_range = min(inst_range, inst_allowed_range)

        pos_min = round(inst_min + (inst_range - inst_allowed_range) / 2)
        pos_max = round(inst_max - (inst_range - inst_allowed_range) / 2)
        assert pos_min <= pos_max
        return pos_min, pos_max, inst_min, inst_max

    def _is_location_filtering(self):
        return self._params.location_filter_ratio < 1.0

    def is_train_test_ended(self) -> bool:
        if self._params.dataset_config != DatasetConfig.TRAIN_TEST:
            print('warning: the dataset is not configured to TRAIN->TEST, ' +
                  ' therefore it will not indicate the simulation end correctly')
            return False
        return self._presented >= len(self._train_label_indexes) + len(self._test_label_indexes)

    def _copy_to_outputs_from(self, pos: int, training: bool):
        if training:
            images = self._train_images
            labels = self._train_labels
            label_indexes = self._train_label_indexes
            instance_ids = self._train_instance_ids

            if self._labels_hidden:
                self.last_label.copy_(self.hidden_label)
            else:
                self.last_label.copy_(labels[label_indexes[pos]])
        else:
            images = self._test_images
            labels = self._test_labels
            label_indexes = self._test_label_indexes
            instance_ids = self._test_instance_ids
            self.last_label.copy_(self.hidden_label)

        image = images[label_indexes[pos]]
        if self._save_memory:
            image = image.to(self._device)
        self.last_image.copy_(image.type(self._float_dtype) / 255.0)
        self.last_truth.copy_(labels[label_indexes[pos]])
        self.last_instance_id.copy_(instance_ids[pos])
        self.testing_phase_indicator.copy_(
            torch.tensor([0.0 if training else 1.0], dtype=self._float_dtype, device=self._device))

    def _filter_labels(self):

        if self._params.class_filter is None:
            self._train_label_indexes = list(range(self._train_labels.shape[0]))
            self._test_label_indexes = list(range(self._test_labels.shape[0]))
        else:
            # one-hot to indices
            _, train_class_ids = self._train_labels.max(1)
            _, test_class_ids = self._test_labels.max(1)

            # indices to list
            train_labels = train_class_ids.long().tolist()
            test_labels = test_class_ids.long().tolist()

            # extract just indices with required class labels
            self._train_label_indexes = DatasetSeObjectsUnit._extract_classes(train_labels, self._params.class_filter)
            self._test_label_indexes = DatasetSeObjectsUnit._extract_classes(test_labels, self._params.class_filter)

    @staticmethod
    def _extract_class(labels: List[int], class_index: int):
        """Filter the positions in the dataset for a corresponding class_index.

        Args:
            class_index: id of the class

        Returns:
            Indexes in the dataset.
        """
        class_ids = [i for i, label in enumerate(labels) if label == class_index]
        return class_ids

    @staticmethod
    def _extract_classes(labels: List[int], class_indexes: List[int]) -> List[int]:
        """Extract examples_per_class samples for each class in the class_indexes (concatenate the results).

        Args:
            class_indexes: list of classes of interest

        Returns:
            Set of all filtered positions in the dataset.
        """
        indexes = set()
        for class_id in class_indexes:
            filtered = DatasetSeObjectsUnit._extract_class(labels, class_id)
            indexes = set.union(indexes, set(filtered))

        result = list(indexes)
        result.sort()
        return result


class DatasetSeObjectsOutputs(SpaceEngineersConnectorOutputs):
    """All outputs of the DatasetSeObjectsNode.

    image_output: width by height RGB image.

    task_to_agent_label: label of the object in the Task0, revealed only in the training phase.
    task_to_agent_label_ground_truth: label of the object in the Task0, revealed always.

    metadata_testing_phase: information if the sample is from the training/testing phase of the Task0.
    metadata_task_id: information about the task id.

    reward_output: blank output.

    task_to_agent_location: blank output.
    task_to_agent_location_int: blank output.
    task_to_agent_location_one_hot: blank output.


    task_to_agent_location_target: blank output.
    task_to_agent_target: blank output.
    task_to_agent_location_target_one_hot: blank output.


    metadata_task_instance_id: blank output.
    metadata_task_status: blank output.
    metadata_task_instance_status: blank output.
    """

    def __init__(self, owner):
        super().__init__(owner)
        self.task_to_agent_label_ground_truth = self.create("Label_ground_truth")

    def prepare_slots(self, unit: DatasetSeObjectsUnit):
        self.image_output.tensor = unit.last_image
        self.task_to_agent_label.tensor = unit.last_label
        self.task_to_agent_label_ground_truth.tensor = unit.last_truth

        self.metadata_testing_phase.tensor = unit.testing_phase_indicator
        self.metadata_task_id.tensor = unit.task_id_const

        # potentially might be used here
        self.reward_output.tensor = unit.unused_output

        # not used here
        self.task_to_agent_location.tensor = unit.unused_output
        self.task_to_agent_location_int.tensor = unit.unused_output
        self.task_to_agent_location_one_hot.tensor = unit.unused_output

        self.task_to_agent_location_target.tensor = unit.unused_output
        self.task_to_agent_target.tensor = unit.unused_output
        self.task_to_agent_location_target_one_hot.tensor = unit.unused_output

        self.metadata_task_instance_id.tensor = unit.last_instance_id
        self.metadata_task_status.tensor = unit.unused_output
        self.metadata_task_instance_status.tensor = unit.unused_output


class DatasetSeObjectsInternals(MemoryBlocks):
    """Introduced as a temporary solution to the serialization.

    Position in the training set should be serialized if stored in the MemoryBlock.
    """

    def __init__(self, owner):
        super().__init__(owner)
        self.training_pos = self.create("Training_pos")

    def prepare_slots(self, unit: DatasetSeObjectsUnit):
        self.training_pos.tensor = unit.training_pos


class DatasetSeObjectsNode(WorkerNodeWithInternalsBase[SpaceEngineersConnectorInputs,
                                                       DatasetSeObjectsInternals,
                                                       DatasetSeObjectsOutputs]):
    """Generates a samples for Task1 from an external dataset.

    Parameters of the samples generation are maintained by DatasetSeObjectsParams class.
    In case of random samples generation it is possible to set the seed value.
    """

    _unit: DatasetSeObjectsUnit
    _seed: int
    _params: DatasetSeObjectsParams

    def __init__(self, params: DatasetSeObjectsParams, seed: int = None, name: str = "DatasetSEObjects"):
        super().__init__(name=name,
                         outputs=DatasetSeObjectsOutputs(self),
                         memory_blocks=DatasetSeObjectsInternals(self),
                         inputs=SpaceEngineersConnectorInputs(self))
        self._seed = seed
        self._params = params.clone()

    def _create_unit(self, creator: TensorCreator):
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)

        return DatasetSeObjectsUnit(creator, self._params, random)

    @property
    def seed(self) -> int:
        return self._params.seed

    @seed.setter
    def seed(self, value: int):
        validate_positive_int(value)
        self._params.seed = value

    @property
    def class_filter(self) -> Optional[List[int]]:
        if self._params.class_filter is None:
            self._params.class_filter = list(range(20))
        return self._params.class_filter

    @class_filter.setter
    def class_filter(self, value: Optional[List[int]]):
        self._params.class_filter = value

    @property
    def random_order(self) -> bool:
        return self._params.random_order

    @random_order.setter
    def random_order(self, value: bool):
        self._params.random_order = value

    @property
    def save_gpu_memory(self) -> bool:
        return self._params.save_gpu_memory

    @save_gpu_memory.setter
    def save_gpu_memory(self, value: bool):
        self._params.save_gpu_memory = value

    @property
    def location_filter_ratio(self) -> float:
        return self._params.location_filter_ratio

    @location_filter_ratio.setter
    def location_filter_ratio(self, value: float):
        validate_positive_float(value)
        if value > 1.0:
            raise FailedValidationException(f'location_filter_ratio expects values from (0,1>, but got {value}')
        self._params.location_filter_ratio = value

    @property
    def dataset_size(self) -> SeDatasetSize:
        return self._params.dataset_size

    @dataset_size.setter
    def dataset_size(self, value: SeDatasetSize):
        self._params.dataset_size = value

    @property
    def dataset_config(self) -> DatasetConfig:
        return self._params.dataset_config

    @dataset_config.setter
    def dataset_config(self, value: DatasetConfig):
        self._params.dataset_config = value

    @property
    def switch_train_resets_train_pos(self) -> bool:
        return self._params.switch_train_resets_train_pos

    @switch_train_resets_train_pos.setter
    def switch_train_resets_train_pos(self, value: bool):
        self._params.switch_train_resets_train_pos = value

    @property
    def is_hide_labels(self) -> bool:
        return self._params.is_hide_labels

    @is_hide_labels.setter
    def is_hide_labels(self, value: bool):
        self._params.is_hide_labels = value
        self._unit._params.is_hide_labels = value
        self._unit._labels_hidden = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""
        return [
            self._prop_builder.auto('Seed', type(self).seed),
            self._prop_builder.auto('Class filter', type(self).class_filter),
            self._prop_builder.auto('Random order', type(self).random_order),
            self._prop_builder.auto('Save gpu memory', type(self).save_gpu_memory),
            self._prop_builder.auto('Location filter ration', type(self).location_filter_ratio),
            self._prop_builder.auto('Dataset size', type(self).dataset_size),
            self._prop_builder.auto('Dataset config', type(self).dataset_config),
            self._prop_builder.auto('Switch training resets train pos ', type(self).switch_train_resets_train_pos),
            self._prop_builder.auto('Hide labels', type(self).is_hide_labels)
        ]

    def _step(self):
        self._unit.step()

    def validate(self):
        if not 0 < self._params.location_filter_ratio <= 1.0:
            raise NodeValidationException(f'validation error: location_filter_ratio expected to be from (0,1>')

    @staticmethod
    def label_size():
        # TODO this is not nice, but this has to be called before the unit is created\
        # (and therefore before we determine num labels from the dataset)
        return DatasetSeObjectsUnit.NUM_LABELS

    def is_training(self):
        # before the simulation:
        if self._unit is None:
            return (self._params.dataset_config == DatasetConfig.TRAIN_ONLY or
                    self._params.dataset_config == DatasetConfig.TRAIN_TEST)
        # during the simulation
        return self._unit._is_training

    def switch_training(self, training_on: bool, just_hide_labels: bool = False):
        """Can switch between training and testing.

        Args:
            training_on: whether to enable/disable training.
            just_hide_labels: if true, the data are not taken from the testing set, just the label tensor is hidden
        """
        logger.info(f'SeObjectsDataset: switching from {self._params.dataset_config} to training_on: {training_on}')

        if just_hide_labels:
            if self._unit is not None:
                self._unit._labels_hidden = not training_on
        else:
            if training_on:
                self._params.dataset_config = DatasetConfig.TRAIN_ONLY
                self._unit._is_training = True
                if self._params.switch_train_resets_train_pos:
                    assert self._unit is not None
                    # the first thing before step is loading pos from this tensor (in testing)
                    self._unit.training_pos[0] = -1
                    self._unit._pos = -1  # just for the purpose of testing
            else:
                self._params.dataset_config = DatasetConfig.TEST_ONLY
                assert self._unit is not None
                self._unit._pos = -1
                self._unit._is_training = False

    def is_train_test_ended(self) -> bool:
        """Return True if one train/test cycle ended"""
        if self._unit is not None:
            return self._unit.is_train_test_ended()
        else:
            return False
