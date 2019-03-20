from dataclasses import dataclass, field
from enum import Enum, auto
from numpy.random.mtrand import RandomState
from typing import List, Optional

from torchsim.core.datasets.alphabet.alphabet import AlphabetGenerator
from torchsim.core.graph.node_base import EmptyInputs, NodeValidationException
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeWithInternalsBase
from torchsim.core.memory.tensor_creator import TensorCreator
import torch
import numpy as np

from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.utils.sequence_generator import SequenceGenerator
from torchsim.core.utils.tensor_utils import id_to_one_hot
from torchsim.gui.validators import validate_predicate
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed


class DatasetAlphabetMode(Enum):
    SEQUENCE_PROBS = auto()


@dataclass
class DatasetAlphabetSequenceProbsModeParams(ParamsBase):
    seqs: List[str] = field(default_factory=list)
    transition_probs: Optional[np.array] = None


@dataclass
class DatasetAlphabetParams(ParamsBase):
    symbols: Optional[str]
    padding_right: int = 0
    mode: DatasetAlphabetMode = DatasetAlphabetMode.SEQUENCE_PROBS
    sequence_probs: Optional[DatasetAlphabetSequenceProbsModeParams] = None


class DatasetAlphabetUnit(Unit):
    output_data: torch.Tensor
    output_sequence_id: torch.Tensor
    output_label: torch.Tensor
    seq: SequenceGenerator
    all_symbols: torch.Tensor
    _current = 0

    def __init__(self, creator: TensorCreator, params: DatasetAlphabetParams, random: Optional[RandomState] = None):
        super().__init__(creator.device)
        self._validate_params(params)

        random = random or np.random.RandomState()

        # Generate all symbols
        generator = AlphabetGenerator(params.padding_right)
        all_symbols = generator.create_symbols(params.symbols)
        self.all_symbols = creator.zeros_like(all_symbols)
        self.all_symbols.copy_(all_symbols.to(creator.device))

        # Create output tensors
        shape = list(self.all_symbols.shape)
        self.output_data = creator.zeros(shape[1:], device=creator.device)
        self.output_label = creator.zeros(1, dtype=torch.int64, device=creator.device)
        self.output_sequence_id = creator.zeros(1, dtype=torch.int64, device=creator.device)
        self.output_sequence_id_one_hot = creator.zeros((1, len(params.sequence_probs.seqs)), dtype=self._float_dtype,
                                                        device=creator.device)

        if params.mode == DatasetAlphabetMode.SEQUENCE_PROBS:
            seqs = [self.convert_string_to_positions(params.symbols, seq) for seq in params.sequence_probs.seqs]
            transition_probs = params.sequence_probs.transition_probs or SequenceGenerator.default_transition_probs(
                seqs)
            self.seq = SequenceGenerator(seqs, transition_probs, random=random)
            self._current = next(self.seq)

        self._n_symbols = shape[0]

    def _validate_params(self, params: DatasetAlphabetParams):
        if params.mode == DatasetAlphabetMode.SEQUENCE_PROBS:
            validate_predicate(lambda: params.sequence_probs is not None,
                               'params.sequence_probs must be set when params.mode == SEQUENCE_PROBS')

    @property
    def transition_probs(self) -> np.array:
        return self.seq.transition_probs

    @transition_probs.setter
    def transition_probs(self, value: np.array):
        self.seq.transition_probs = value

    @staticmethod
    def convert_string_to_positions(symbols: str, text: str) -> List[int]:
        return [symbols.index(s) for s in text]

    def step(self):
        self.output_data.copy_(self.all_symbols[self._current])
        self.output_label[0] = self._current
        self.output_sequence_id[0] = self.seq.current_sequence_id
        self.output_sequence_id_one_hot.copy_(
            id_to_one_hot(self.output_sequence_id, self.output_sequence_id_one_hot.shape[1]))
        self._current = next(self.seq)


class DatasetAlphabetInternals(MemoryBlocks):
    all_symbols: MemoryBlock

    def __init__(self, owner):
        super().__init__(owner)
        self.all_symbols = self.create("All symbols")

    def prepare_slots(self, unit: DatasetAlphabetUnit):
        self.all_symbols.tensor = unit.all_symbols


class DatasetAlphabetOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output symbol")
        self.label = self.create("Label")
        self.sequence_id = self.create("Sequence ID")
        self.sequence_id_one_hot = self.create("Sequence ID one hot")

    def prepare_slots(self, unit: DatasetAlphabetUnit):
        self.output.tensor = unit.output_data
        self.label.tensor = unit.output_label
        self.sequence_id.tensor = unit.output_sequence_id
        self.sequence_id_one_hot.tensor = unit.output_sequence_id_one_hot


class DatasetAlphabetNode(WorkerNodeWithInternalsBase[EmptyInputs, DatasetAlphabetInternals, DatasetAlphabetOutputs]):
    _seed: Optional[int]
    _params: DatasetAlphabetParams
    _unit: DatasetAlphabetUnit

    def __init__(self, params: DatasetAlphabetParams, seed: Optional[int] = None, name="DatasetAlphabet"):
        super().__init__(name=name, outputs=DatasetAlphabetOutputs(self), memory_blocks=DatasetAlphabetInternals(self))
        self._params = params.clone()
        self._seed = seed

    def _create_unit(self, creator: TensorCreator) -> DatasetAlphabetUnit:
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)
        return DatasetAlphabetUnit(creator, self._params, random)

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



