import numpy as np
from torchsim.core.graph import logging, Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams, SpatialPoolerParams, TemporalPoolerParams
from torchsim.core.nodes import ExpertFlockNode, UnsqueezeNode, ExpandNode
from torchsim.core.nodes.dataset_alphabet_node import DatasetAlphabetNode, DatasetAlphabetParams, \
    DatasetAlphabetSequenceProbsModeParams

logger = logging.getLogger(__name__)


class SymbolicInputWordsTopology(Topology):
    """
    Long words utilizing context
    """
    _dataset_node: DatasetAlphabetNode
    _expert_child_node: ExpertFlockNode
    # _step_count: int = 0

    def __init__(self):
        super().__init__('cuda')
        self.toploogy_dataset_test()

    def toploogy_dataset_test(self):
        dataset_params = DatasetAlphabetParams(symbols="abcdefghijklmnopqrstuvwxyz0123456789", padding_right=1,
                                               sequence_probs=DatasetAlphabetSequenceProbsModeParams(
                                                   # seqs=['abc', '123', '456789', '468'],
                                                   seqs=['abc123def', 'ghi123jkl'],
                                                   # transition_probs=[[0.7, 0.3], [0.3, 0.7], ]
                                               ))
        dataset_node = DatasetAlphabetNode(
            dataset_params
        )
        flock_size = 1
        # parent_cluster_centers = len(dataset_params.sequence_probs.seqs)
        parent_cluster_centers = 5  # len(dataset_params.sequence_probs.seqs)
        parent2_cluster_centers = 2  # len(dataset_params.sequence_probs.seqs)

        unsqueeze_node_child = UnsqueezeNode(0)
        unsqueeze_node_sequence_id = UnsqueezeNode(0)
        expand_node_child = ExpandNode(0, flock_size)
        expand_node_sequence_id = ExpandNode(0, flock_size)

        child_cluster_centers = len(dataset_params.symbols)
        expert_node_child = ExpertFlockNode(
            ExpertParams(flock_size=flock_size,
                         n_cluster_centers=child_cluster_centers,
                         spatial=SpatialPoolerParams(
                             enable_learning=False
                         ),
                         temporal=TemporalPoolerParams(
                             # incoming_context_size=parent_cluster_centers,
                             n_frequent_seqs=50,
                             seq_length=3,
                             seq_lookahead=1,
                             # n_providers=2
                         )
                         )
        )

        expert_node_parent = ExpertFlockNode(
            ExpertParams(flock_size=flock_size,
                         n_cluster_centers=parent_cluster_centers,
                         spatial=SpatialPoolerParams(
                         ),
                         temporal=TemporalPoolerParams(
                             incoming_context_size=4,
                             n_frequent_seqs=50,
                             seq_length=3,
                             seq_lookahead=1

                         )
                         )
        )
        expert_node_parent2 = ExpertFlockNode(
            ExpertParams(flock_size=flock_size,
                         n_cluster_centers=parent2_cluster_centers,
                         spatial=SpatialPoolerParams(
                         ),
                         temporal=TemporalPoolerParams(
                             incoming_context_size=4,
                             n_frequent_seqs=50,
                             seq_length=3,
                             seq_lookahead=1

                         )
                         )
        )

        expert_node_sequence_id = ExpertFlockNode(
            ExpertParams(flock_size=flock_size,
                         n_cluster_centers=2,
                         spatial=SpatialPoolerParams(
                         ))
        )

        self.add_node(dataset_node)
        self.add_node(unsqueeze_node_child)
        # self.add_node(unsqueeze_node_sequence_id)
        self.add_node(expand_node_child)
        # self.add_node(expand_node_sequence_id)
        self.add_node(expert_node_child)
        self.add_node(expert_node_parent)
        self.add_node(expert_node_parent2)
        # self.add_node(expert_node_sequence_id)

        Connector.connect(dataset_node.outputs.output, unsqueeze_node_child.inputs.input)
        Connector.connect(unsqueeze_node_child.outputs.output, expand_node_child.inputs.input)
        Connector.connect(expand_node_child.outputs.output, expert_node_child.inputs.sp.data_input)
        # Connector.connect(dataset_node.outputs.sequence_id, unsqueeze_node_sequence_id.inputs.input)
        # Connector.connect(unsqueeze_node_sequence_id.outputs.output, expand_node_sequence_id.inputs.input)
        # Connector.connect(expand_node_sequence_id.outputs.output, expert_node_sequence_id.inputs.sp.data_input)

        Connector.connect(expert_node_child.outputs.tp.projection_outputs, expert_node_parent.inputs.sp.data_input)
        # Parent context
        Connector.connect(expert_node_parent.outputs.output_context, expert_node_child.inputs.tp.context_input,
                          is_backward=True)

        # Parent 2
        Connector.connect(expert_node_parent.outputs.tp.projection_outputs, expert_node_parent2.inputs.sp.data_input)
        # Connector.connect(expert_node_parent2.outputs.output_context, expert_node_child.inputs.tp.context_input,
        #                   is_backward=True)


        # Sequence id context
        # Connector.connect(expert_node_sequence_id.outputs.output_context, expert_node_child.inputs.tp.context_input,
        #                   is_backward=True)
        self._dataset_node = dataset_node
        self._expert_child_node = expert_node_child
        self._flock_size = flock_size

    def init_sp_clusters(self):
        # self._expert_child_node.
        symbols = self._dataset_node.memory_blocks.all_symbols.tensor
        cluster_centers = symbols.unsqueeze(0).expand((self._flock_size, *symbols.shape))
        self._expert_child_node.memory_blocks.sp.cluster_centers.tensor.copy_(cluster_centers)

    # def step(self):
    #     super().step()
    #     self._step_count += 1
    #     # logger.info(f"Step {self._step_count}")
    #     if self._step_count == 1:
    #         self.setup_sp_clusters()
    #         logger.info("Symbols injected to SP")

    # @property
    # def dataset_symbols(self) -> torch.Tensor:
    #     return self._dataset_node.memory_blocks.all_symbols.tensor
