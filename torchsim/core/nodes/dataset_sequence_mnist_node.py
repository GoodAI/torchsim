import ast
import logging
from copy import deepcopy

import numpy as np

from torchsim.core.datasets.mnist import DatasetMNIST
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.nodes.dataset_mnist_node import DatasetMNISTParams, DatasetSequenceMNISTNodeParams, \
    DatasetMNISTOutputs, DatasetMNISTUnit
from torchsim.gui.observables import disable_on_runtime
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.validators import *
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed

logger = logging.getLogger(__name__)


class DatasetSequenceMNISTOutputs(DatasetMNISTOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.sequence_id = self.create("Sequence ID")

    def prepare_slots(self, unit: DatasetMNISTUnit):
        super().prepare_slots(unit)
        self.sequence_id.tensor = unit.output_sequence_id


class DatasetSequenceMNISTNode(WorkerNodeBase[EmptyInputs, DatasetSequenceMNISTOutputs]):
    """Presents the MNIST images (and labels) in specified sequences.

    Properties of the sequences can be setup by the SequenceParams.
    Sequence is defined by the labels in the sequence and transition probability matrix.
    Further configuration can be added by the DatasetMNISTParams (e.g. bitmaps_per_class),
    DatasetMNISTParams/class_filter is ignored in this Node (class filter parsed from the SequenceParams).

    This Node is not anywhere and not thoroughly tested.
    """

    def __init__(self, params: DatasetMNISTParams, seq_params: DatasetSequenceMNISTNodeParams,
                 dataset: Optional[DatasetMNIST] = None, seed: int = None,
                 name="DatasetSequenceMNISTNode"):

        super().__init__(name=name, outputs=DatasetSequenceMNISTOutputs(self))
        self._params = params.clone()
        self._seq_params = seq_params
        self._dataset = dataset or DatasetMNIST()
        self._seed = seed

    def _create_unit(self, creator: TensorCreator):
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)

        return DatasetMNISTUnit(creator,
                                self._dataset,
                                deepcopy(self._params),
                                random=random,
                                seq_params=deepcopy(self._seq_params))

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
    def examples_per_class(self) -> Optional[int]:
        return self._params.examples_per_class

    @examples_per_class.setter
    def examples_per_class(self, value: Optional[int]):
        validate_positive_optional_int(value)
        self._params.examples_per_class = value

    @property
    def seqs(self) -> List[List[int]]:
        return self._seq_params.seqs

    @seqs.setter
    def seqs(self, seqs: str):
        try:
            value = ast.literal_eval(seqs)
            validate_list_list_int(value)
            self._seq_params.seqs = value
        except (ValueError, SyntaxError, TypeError):
            raise FailedValidationException('List[List[int]] object expected (e.g. [[1,2],[3]])')

    @property
    def custom_transition_probs(self):
        return self._seq_params.transition_probs

    @custom_transition_probs.setter
    def custom_transition_probs(self, value):
        if value is None:
            self._seq_params.transition_probs = None
        else:
            try:
                parsed_value = ast.literal_eval(value)
                validate_list_list_float_or_int(parsed_value)
                self._seq_params.transition_probs = np.array(parsed_value)
            except (ValueError, SyntaxError, TypeError):
                raise FailedValidationException('List[List[float]] object expected (e.g. [[1.5,2],[3]])')

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""

        ret_list = super().get_properties()
        ret_list.extend([
            self._prop_builder.prop('Sequences', type(self).seqs, lambda x: x, str,
                                    resolve_strategy=disable_on_runtime),
            self._prop_builder.prop('Custom transition probs', type(self).custom_transition_probs, parser=lambda x: x,
                                    formatter=lambda v: str(v.tolist()) if v is not None else None,
                                    resolve_strategy=disable_on_runtime, optional=True),
            self._prop_builder.auto('Examples per class', type(self).examples_per_class,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Random order', type(self).random_order, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('One-hot labels', type(self).one_hot_labels, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Seed', type(self).seed, edit_strategy=disable_on_runtime),
        ])

        return ret_list

    def _step(self):
        self._unit.step()
