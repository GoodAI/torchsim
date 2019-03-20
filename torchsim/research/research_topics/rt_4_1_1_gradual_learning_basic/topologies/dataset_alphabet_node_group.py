import itertools
import torch
import numpy as np

import logging
from dataclasses import dataclass
from typing import List

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.node_group import NodeGroupBase, GroupOutputs
from torchsim.core.models.expert_params import ExpertParams, SpatialPoolerParams
from torchsim.core.nodes import UnsqueezeNode, SpatialPoolerFlockNode, ExpandNode
from torchsim.core.nodes.dataset_alphabet_node import DatasetAlphabetSequenceProbsModeParams, DatasetAlphabetNode, \
    DatasetAlphabetParams
from torchsim.core.nodes.multi_dataset_alphabet_node import MultiDatasetAlphabetNode

logger = logging.getLogger(__name__)


@dataclass
class DatasetAlphabetNodeGroupParams:
    flock_size: int
    symbols: str
    seq_length: int
    seq_count: int
    seq_repeat: int


class DatasetAlphabetNodeGroupOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")
        self.sequence_id_one_hot = self.create("Sequence ID one hot")
        self.scalar_sequence_ids = self.create("Scalar sequence ID")


class DatasetAlphabetNodeGroup(NodeGroupBase[EmptyInputs, DatasetAlphabetNodeGroupOutputs]):
    def __init__(self,
                 params: DatasetAlphabetNodeGroupParams,
                 name: str = "DatasetAlphabetNodeGroup"):
        super().__init__(name, inputs=EmptyInputs(self), outputs=DatasetAlphabetNodeGroupOutputs(self))
        self._params = params

        dataset_node = self.create_node_dataset()

        sp_dataset_node = SpatialPoolerFlockNode(
            ExpertParams(flock_size=self._params.flock_size,
                         n_cluster_centers=len(self._params.symbols),
                         spatial=SpatialPoolerParams(
                             enable_learning=False
                         ),
                         ),
            name="SP_dataset"
        )

        # Dataset nodes
        self.add_node(dataset_node)
        self.add_node(sp_dataset_node)
        # Connect data output
        Connector.connect(dataset_node.outputs.outputs, sp_dataset_node.inputs.sp.data_input)
        # Connect sequence_id output
        Connector.connect(dataset_node.outputs.sequence_ids_one_hot, self.outputs.sequence_id_one_hot.input)
        Connector.connect(sp_dataset_node.outputs.sp.forward_clusters, self.outputs.output.input)
        Connector.connect(dataset_node.outputs.sequence_ids, self.outputs.scalar_sequence_ids.input)

        self._dataset_node = dataset_node
        self._sp_dataset_node = sp_dataset_node

    def create_node_dataset(self):
        def generate_sequence(symbols: str, count: int, skip: int):
            """Generate sequence of length `count` and stride `skip` from `symbols` - symbols are repeated when needed"""
            repeating_symbols = itertools.cycle(symbols)
            result = []
            for i in range(count):
                result.append(next(repeating_symbols))
                for s in range(skip):
                    next(repeating_symbols)
            return ''.join(result)

        def count_unique_symbols(sequences: List[str]) -> int:
            symbols = set()
            for seq in sequences:
                for symbol in seq:
                    symbols.add(symbol)
            return len(symbols)

        seqs = [str(generate_sequence(self._params.symbols, self._params.seq_length, i)) for i in
                range(self._params.seq_count)]

        # duplicate sequences
        for s in list(seqs):
            seqs.append(s)
        # make sequences longer (repeat)
        seqs = [s * self._params.seq_repeat for s in seqs]

        dataset_params = DatasetAlphabetParams(symbols=self._params.symbols, padding_right=1,
                                               sequence_probs=DatasetAlphabetSequenceProbsModeParams(seqs=seqs))
        return MultiDatasetAlphabetNode(dataset_params, n_worlds=self._params.flock_size)

    def init_sp_clusters(self):
        cluster_centers = self.symbols.unsqueeze(0).expand((self._params.flock_size, *self.symbols.shape))
        self._sp_dataset_node.memory_blocks.sp.cluster_centers.tensor.copy_(cluster_centers)

    def set_sequences_filter(self, enabled_sequences: List[bool]):
        self._dataset_node.transition_probs = self._compute_sequences_probs(enabled_sequences)

    @staticmethod
    def _compute_sequences_probs(enabled_sequences: List[bool]) -> np.array:
        enabled_sequences_count = len(list(filter(None, enabled_sequences)))
        line = []
        for enabled in enabled_sequences:
            line.append(1 / enabled_sequences_count if enabled else 0)
        probs = np.array([line] * 6)
        logger.debug('probs: ' + str(probs))
        return probs

    @property
    def symbols(self) -> torch.Tensor:
        return self._dataset_node.memory_blocks.all_symbols.tensor
