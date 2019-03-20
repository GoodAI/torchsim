import logging
from dataclasses import dataclass
from typing import List

from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.nodes.dataset_se_objects_node import DatasetSeObjectsParams, DatasetSeObjectsUnit, DatasetSeObjectsNode
from torchsim.core.persistence.loader import Loader
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed

logger = logging.getLogger(__name__)

@dataclass
class SeObjectsTaskPhaseParams(ParamsBase):
    """Class for configuring a single phase of the gradual task."""
    class_filter: List[int]
    location_filter: float
    is_training: bool
    num_steps: int

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """Converts to string.

        The returned string is used for constructing file names.
        """
        return '_'.join([str(self.class_filter), str(self.location_filter), str(self.is_training), str(self.num_steps)])

@dataclass
class PhasedSeObjectsTaskParams(ParamsBase):
    """Class used for configuring the whole task."""
    phases: List[SeObjectsTaskPhaseParams]
    dataset_params: DatasetSeObjectsParams


class PhasedSeObjectsTaskUnit(DatasetSeObjectsUnit):
    """Adds switching between learning and testing phases and different class filtering."""
    _counter: int
    _current_phase_index: int
    _phase_is_training: bool

    def __init__(self, creator: TensorCreator, params: PhasedSeObjectsTaskParams, random):
        super().__init__(creator, params.dataset_params, random)
        self._phases = params.phases
        self._current_phase_index = -1
        self._phase_is_training = True
        self._stored_train_instance_ids = self._train_instance_ids  # hack
        self._stored_train_images = self._train_images  # hack
        self._stored_train_labels = self._train_labels  # hack
        self._set_next_phase_params()
        self._stored_training_pos = 0

    def load(self, parent_loader: Loader):
        pass

    def _set_next_phase_params(self):
        self._counter = 0
        self._current_phase_index += 1
        if self._current_phase_index >= len(self._phases):
            print("Warning: end of task data, starting again from the beginning")
            self._current_phase_index = 0
        self._params.class_filter = self._phases[self._current_phase_index].class_filter
        self._params.location_filter_ratio = self._phases[self._current_phase_index].location_filter
        if self._phase_is_training:
            self._stored_training_pos = self._pos

        self._train_instance_ids = self._stored_train_instance_ids  # part of the hack
        self._train_images = self._stored_train_images
        self._train_labels = self._stored_train_labels

        self._reset_indexes_and_filtering()  # TODO: check this

        self._phase_is_training = self._phases[self._current_phase_index].is_training
        if self._phase_is_training:
            self._pos = self._stored_training_pos
            self.training_pos[0] = self._pos  # write to the tensor too

        if not self._phase_is_training:  # hack!
            self._train_label_indexes = self._test_label_indexes
            self._train_instance_ids = self._test_instance_ids
            self._train_labels = self._test_labels
            self._train_images = self._test_images

    def step(self):
        if self._counter >= self._phases[self._current_phase_index].num_steps:
            self._set_next_phase_params()

        self._counter += 1
        super().step()

    def switch_training(self, training_on: bool, just_hide_labels: bool = False):
        """Can switch between training and testing.

        Args:
            training_on: whether to enable/disable training.
            just_hide_labels: if true, the data are not taken from the testing set, just the label tensor is hidden
        """
        logger.warning(f'SeObjectsDataset: switching from {self._params.dataset_config} to training_on: {training_on}')

        if just_hide_labels:
            self._labels_hidden = not training_on

        # skip the base class code, we're switching the training/testing ourselves already in _set_next_phase_params

    def phase_is_training(self):
        return self._phase_is_training


class PhasedSeObjectsTaskNode(DatasetSeObjectsNode):
    """Node based on Task0 dataset node, changing provided outputs (images from Task 0) in phases.

    The node is configured by PhasedSeObjectsTaskParams: these allow the node to change outputs from phase to phase.
    Every phase can configure the classes of objects that appear on the output and their location ratios.
    Every phase lasts for a pre-configured number of simulation steps.

    For example, in the first phase, the node can provide data of classes 1 and 2,
                 in the second phase, the node can provide data of classes 2 and 3, with location ratio shrunk to 0.2
    """

    _unit: PhasedSeObjectsTaskUnit
    _seed: int

    def __init__(self, params: PhasedSeObjectsTaskParams, seed: int = None):
        super().__init__(params=params.dataset_params, name="PhasedSeObjectsTask", seed=seed)
        self._phased_params = params.clone()
        self._seed = seed

    def _create_unit(self, creator: TensorCreator):
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)

        return PhasedSeObjectsTaskUnit(creator, self._phased_params, random)

    def _step(self):
        self._unit.step()

    def is_training(self):
        # before the simulation:
        if self._unit is None:
            return True
        # during the simulation
        return self._unit._phase_is_training

    def switch_training(self, training_on: bool, just_hide_labels: bool = False):
        """Can switch between training and testing.

        Args:
            training_on: whether to enable/disable training.
            just_hide_labels: if true, the data are not taken from the testing set, just the label tensor is hidden
        """

        if self._unit is not None:
            self._unit._labels_hidden = not training_on

    def is_train_test_ended(self) -> bool:
        """Return True if one train/test cycle ended."""
        return False  # currently not used
