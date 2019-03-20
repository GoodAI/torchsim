from typing import List, Optional

import numpy as np
from numpy.random.mtrand import RandomState

import torch
from torchsim.core.graph.node_base import EmptyInputs, NodeValidationException
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeWithInternalsBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.nodes.dataset_alphabet_node import DatasetAlphabetUnit, DatasetAlphabetParams, DatasetAlphabetMode
from torchsim.gui.validators import validate_predicate
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed


class MultiDatasetAlphabetUnit(Unit):
    output_data: torch.Tensor
    output_label: torch.Tensor
    output_sequence_id: torch.Tensor
    output_sequence_id_one_hot: torch.Tensor
    all_symbols: torch.Tensor

    n_worlds: int
    _units: List[DatasetAlphabetUnit]
    _first_unit: DatasetAlphabetUnit

    def __init__(self, creator: TensorCreator, params: DatasetAlphabetParams, n_worlds: int,
                 random: Optional[RandomState] = None):
        super().__init__(creator.device)
        self._validate_params(params)
        self.n_worlds = n_worlds

        self._units = [DatasetAlphabetUnit(creator, params.clone(), random) for _ in range(self.n_worlds)]

        def stacked(tensor):
            size = [self.n_worlds] + list(tensor.shape)
            return creator.zeros(size, dtype=tensor.dtype, device=tensor.device)

        self._first_unit = self._units[0]

        # Create output tensors
        self.output_data = stacked(self._first_unit.output_data)
        self.output_label = stacked(self._first_unit.output_label)
        self.output_sequence_id = stacked(self._first_unit.output_sequence_id)
        self.output_sequence_id_one_hot = stacked(self._first_unit.output_sequence_id_one_hot)
        self.all_symbols = self._first_unit.all_symbols

    def _validate_params(self, params: DatasetAlphabetParams):
        if params.mode == DatasetAlphabetMode.SEQUENCE_PROBS:
            validate_predicate(lambda: params.sequence_probs is not None,
                               'params.sequence_probs must be set when params.mode == SEQUENCE_PROBS')

    @property
    def transition_probs(self) -> np.array:
        return self._first_unit.seq.transition_probs

    @transition_probs.setter
    def transition_probs(self, value: np.array):
        for unit in self._units:
            unit.seq.transition_probs = value

    def step(self):
        for unit in self._units:
            unit.step()

        torch.stack([unit.output_data for unit in self._units], dim=0, out=self.output_data)
        torch.stack([unit.output_label for unit in self._units], dim=0, out=self.output_label)
        torch.stack([unit.output_sequence_id for unit in self._units], dim=0, out=self.output_sequence_id)
        torch.stack([unit.output_sequence_id_one_hot for unit in self._units], dim=0,
                    out=self.output_sequence_id_one_hot)


class MultiDatasetAlphabetInternals(MemoryBlocks):
    all_symbols: MemoryBlock

    def __init__(self, owner):
        super().__init__(owner)
        self.all_symbols = self.create("All symbols")

    def prepare_slots(self, unit: DatasetAlphabetUnit):
        self.all_symbols.tensor = unit.all_symbols


class MultiDatasetAlphabetOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.outputs = self.create("Outputs symbol")
        self.labels = self.create("Labels")
        self.sequence_ids = self.create("Sequence IDs")
        self.sequence_ids_one_hot = self.create("Sequence IDs one hot")

    def prepare_slots(self, unit: MultiDatasetAlphabetUnit):
        self.outputs.tensor = unit.output_data
        self.labels.tensor = unit.output_label
        self.sequence_ids.tensor = unit.output_sequence_id
        self.sequence_ids_one_hot.tensor = unit.output_sequence_id_one_hot.view(unit.n_worlds, -1)


class MultiDatasetAlphabetNode(WorkerNodeWithInternalsBase[EmptyInputs, MultiDatasetAlphabetInternals,
                                                           MultiDatasetAlphabetOutputs]):
    _seed: Optional[int]
    _params: DatasetAlphabetParams
    _unit: MultiDatasetAlphabetUnit
    _n_worlds: int

    def __init__(self, params: DatasetAlphabetParams, seed: Optional[int] = None, n_worlds: int = 1,
                 name: str = "DatasetAlphabet"):
        super().__init__(name=name, outputs=MultiDatasetAlphabetOutputs(self),
                         memory_blocks=MultiDatasetAlphabetInternals(self))
        self._params = params.clone()
        self._seed = seed
        self._n_worlds = n_worlds

    def _create_unit(self, creator: TensorCreator) -> MultiDatasetAlphabetUnit:
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)
        return MultiDatasetAlphabetUnit(creator, self._params, self._n_worlds, random)

    def _step(self):
        self._unit.step()

    def validate(self):
        if self._params.mode == DatasetAlphabetMode.SEQUENCE_PROBS:
            symbols = self._params.symbols
            for seq in self._params.sequence_probs.seqs:
                for s in seq:
                    if symbols.find(s) == -1:
                        raise NodeValidationException(
                            f'Symbol "{s}" in sequence "{seq}" was not found in all symbols "{symbols}". '
                            f'Remove it from the sequence or add id to symbols.')

    @property
    def transition_probs(self) -> np.array:
        return self._unit.transition_probs

    @transition_probs.setter
    def transition_probs(self, value: np.array):
        self._unit.transition_probs = value
