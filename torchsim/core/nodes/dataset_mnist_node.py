import logging
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Generator

import numpy as np
from numpy.random.mtrand import RandomState

import torch
from torchsim.core import FLOAT_TYPE_CPU
from torchsim.core.datasets.mnist import DatasetMNIST
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.utils.sequence_generator import SequenceGenerator
from torchsim.core.utils.tensor_utils import id_to_one_hot
from torchsim.gui.observables import disable_on_runtime
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.validators import *
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed

logger = logging.getLogger(__name__)


@dataclass
class DatasetMNISTParams(ParamsBase):
    """Initialize the MNIST params.

    Fields:
        class_filter: filter the dataset by the class IDs
        examples_per_class: how many examples per class to find, by default, all examples are shown
        random_order: during basic non-sequential sampling, should the order of data be random?
        one_hot_labels: second output should be integer or a one-hot vector?
    """

    class_filter: Optional[List[int]] = None
    examples_per_class: Optional[int] = None
    random_order: bool = True
    one_hot_labels: bool = True


@dataclass
class DatasetSequenceMNISTNodeParams:
    """User-settable parameters for sequences.

    This class represents sequence-related parameters appearing in the user interface.

    Properties are retrieved and set using the data types provided by the GUI code -- for example, lists of
    numbers are represented as text in the GUI. The :parameter_values: property is then used to export the
    parameter values.
    """
    seqs: List[List[int]] = field(default_factory=list)
    transition_probs: Optional[np.array] = None


class DatasetMNISTUnit(Unit):
    _data_seq: Generator[np.ndarray, None, None]
    _params: DatasetMNISTParams
    _seq_params: DatasetSequenceMNISTNodeParams
    _seq: SequenceGenerator

    _random: np.random

    # dataset filtered by the class filter
    _data: torch.Tensor
    _labels: torch.Tensor

    # outputs:
    output_data: torch.Tensor  # used by the Node
    output_label: torch.Tensor
    output_sequence_id: torch.Tensor
    label_tensor: torch.Tensor  # used in the accessor

    _dataset: DatasetMNIST

    def __init__(self,
                 creator: TensorCreator,
                 dataset: DatasetMNIST,
                 params: DatasetMNISTParams,
                 random: RandomState,
                 seq_params: DatasetSequenceMNISTNodeParams = None):
        """A Node which provides MNIST images form the training set (either sequentially or randomly ordered).

        Args:
            params:
            seq_params: if specified, the SequenceGenerator is initialized and is used to generate MNIST sequences.
        """
        super().__init__(creator.device)

        self._params = params
        self._seq_params = seq_params

        self._random = random

        self._dataset = dataset  # instance of the DatasetMNIST to be used

        self._init_node(creator)

    def _init_node(self, creator: TensorCreator):

        device = self._device

        if self._seq_params is not None:
            # each _init_node, also the new SequenceGenerator should be created (reset the current position etc)
            # self._seq = SequenceGenerator.from_params(self._seq_params, random=self._random)
            transition_probs = self._seq_params.transition_probs if self._seq_params.transition_probs is not None \
                else SequenceGenerator.default_transition_probs(self._seq_params.seqs)
            self._seq = SequenceGenerator(self._seq_params.seqs, transition_probs, random=self._random)
            self._update_class_filter_by_sequences()
        else:
            self._seq = None

        self._sanitize_class_filter()

        class_filter = self._params.class_filter

        dataset_result = self._dataset.get_filtered(
            class_filter) if class_filter is not None else self._dataset.get_all()
        self._data = dataset_result.data.type(FLOAT_TYPE_CPU) / 255.0
        self._labels = dataset_result.labels

        self._data_seq = self._get_data_sequence_iterator()

        # init the output tensors
        self.label_tensor = creator.zeros(1, device=device, dtype=torch.int64)
        self.output_sequence_id = creator.zeros(1, device=device, dtype=torch.int64)
        self.output_data = creator.zeros((28, 28), device=device, dtype=self._float_dtype)
        if self._params.one_hot_labels:
            self.output_label = creator.zeros(10, device=device, dtype=self._float_dtype)
        else:
            self.output_label = creator.zeros(1, device=device, dtype=torch.int64)

    def _update_class_filter_by_sequences(self):
        """The class filter should contain just the classes specified in the sequence generator if it is specified."""
        flat_list = []
        for sublist in self._seq_params.seqs:
            for item in sublist:
                flat_list.append(item)

        self._params.class_filter = list(set(flat_list))

    def _sanitize_class_filter(self):
        if self._params.class_filter is None:
            self._params.class_filter = list(range(0, 10))

        self._params.class_filter = [val for val in self._params.class_filter if (0 <= val < 10)]
        if len(self._params.class_filter) == 0:
            self._params.class_filter = list(range(0, 10))

        self._params.class_filter = list(set(self._params.class_filter))

    def _get_data_sequence_iterator(self):
        if self._seq is not None:
            return self._generate_class_seq_data(self._seq)
        elif self._params.random_order:
            return self._generate_random_data()
        else:
            return self._generate_ordered_data()

    def step(self):
        """Returns nothing sets values of the tensors, which are marked as Node outputs (MemoryBlocks)."""

        # the data are held on CPU because they can be quite big and they are sent to GPU just when needed
        data, self.label_tensor = next(self._data_seq)

        # TODO (Time-Optim) the iterator could be changed so that the copying to output tensors is avoided?

        # copy the results to outputs
        self.output_data.copy_(data.to(self._device))
        self.label_tensor = self.label_tensor.to(self._device)

        if self._seq is not None:
            self.output_sequence_id[0] = self._seq.current_sequence_id
        if self._params.one_hot_labels:
            self.output_label.copy_(id_to_one_hot(self.label_tensor, 10))
        else:
            self.output_label.copy_(self.label_tensor)

    @lru_cache(maxsize=10)
    def _extract_class(self, class_index: int):
        """Filter the positions in the dataset for a corresponding class_index.

        Randomly pick just examples_per_class indexes

        Args:
            class_index: id of the class
        Returns:
            indexes in the dataset
        """
        class_ids = [i for i, label in enumerate(self._labels) if label == class_index]
        if self._params.examples_per_class is None:
            return class_ids

        if self._params.examples_per_class >= len(class_ids):
            logger.warning(f'MNIST: params.examples_per_class ({self._params.examples_per_class}) is too big,'
                           f' could find just {len(class_ids)} samples for the class {class_index} ')
        num_requested_samples = min(self._params.examples_per_class, len(class_ids) - 1)
        # pick requested number of randomly chosen bitmaps without repetition
        return self._random.choice(class_ids, num_requested_samples, replace=False)

    def _extract_classes(self, class_indexes: List[int]):
        """Extract examples_per_class samples for each class in the class_indexes (concatenate the results).

        Args:
            class_indexes: list of classes of interest

        Returns:
            set of all filtered positions in the dataset.
        """
        indexes = set()
        for class_id in class_indexes:
            filtered = self._extract_class(class_id)
            indexes = set.union(indexes, set(filtered))

        return list(indexes)

    def _sample_from_class(self, class_index: int):
        choice = self._random.choice(self._extract_class(class_index))
        return self._data[choice], self._labels[choice]

    def _generate_ordered_data(self):
        position = 0
        extracted_ids = self._extract_classes(self._params.class_filter)
        length = len(extracted_ids)
        while True:
            yield self._data[extracted_ids[position]], self._labels[extracted_ids[position]]
            position = (position + 1) % length

    def _generate_random_data(self):
        extracted_ids = self._extract_classes(self._params.class_filter)
        while True:
            yield self._random_choice(extracted_ids)

    def _random_choice(self, extracted_ids: List[int]):
        choice = self._random.choice(len(extracted_ids))
        return self._data[extracted_ids[choice]], self._labels[extracted_ids[choice]]

    def _generate_class_seq_data(self, seq: SequenceGenerator):
        while True:
            yield self._sample_from_class(next(seq))


class DatasetMNISTOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.data = self.create("Output_data")
        self.label = self.create("Output_label")

    def prepare_slots(self, unit: DatasetMNISTUnit):
        self.data.tensor = unit.output_data
        self.label.tensor = unit.output_label


class DatasetMNISTNode(WorkerNodeBase[EmptyInputs, DatasetMNISTOutputs]):
    """Provides the MNIST data as a node in the simulator, provides image and label.

    For configuration see the DatasetMNISTParams documentation.
    """

    _unit: DatasetMNISTUnit

    outputs: DatasetMNISTOutputs
    _params: DatasetMNISTParams
    _dataset: DatasetMNIST
    _seed: Optional[int]

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @seed.setter
    def seed(self, value: Optional[int]):
        validate_positive_optional_int(value)
        self._seed = value

    @property
    def random_order(self) -> bool:
        return self._params.random_order

    @random_order.setter
    def random_order(self, value: bool):
        self._params.random_order = value

    @property
    def one_hot_labels(self) -> bool:
        return self._params.one_hot_labels

    @one_hot_labels.setter
    def one_hot_labels(self, value: bool):
        self._params.one_hot_labels = value

    @property
    def class_filter(self) -> Optional[List[int]]:
        return self._params.class_filter

    @class_filter.setter
    def class_filter(self, value: Optional[List[int]]):
        self._params.class_filter = value

    @property
    def examples_per_class(self) -> Optional[int]:
        return self._params.examples_per_class

    @examples_per_class.setter
    def examples_per_class(self, value: Optional[int]):
        validate_positive_optional_int(value)
        self._params.examples_per_class = value

    def __init__(self, params: DatasetMNISTParams,
                 dataset: DatasetMNIST = None,
                 seed: Optional[int] = None,
                 name="DatasetMNISTNode"):
        super().__init__(name=name, outputs=DatasetMNISTOutputs(self))
        self._params = params.clone()
        self._dataset = dataset or DatasetMNIST()
        self._seed = seed

    def _create_unit(self, creator: TensorCreator):
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)

        return DatasetMNISTUnit(creator,
                                self._dataset,
                                deepcopy(self._params),
                                random=random)

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""

        return [
            self._prop_builder.auto('Class Filter', type(self).class_filter, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Examples per class', type(self).examples_per_class,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Random order', type(self).random_order, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('One-hot labels', type(self).one_hot_labels, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Seed', type(self).seed, edit_strategy=disable_on_runtime)
        ]

    def _step(self):
        self._unit.step()
