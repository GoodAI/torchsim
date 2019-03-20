import numpy as np
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams, SpatialPoolerParams, TemporalPoolerParams
from torchsim.core.nodes import DatasetSequenceMNISTNode, DatasetSequenceMNISTNodeParams, DatasetMNISTParams, \
    ExpertFlockNode, UnsqueezeNode, ExpandNode


class ContextTestTopology(Topology):
    """
    Topology demonstrating correct functionality of context
    """

    def __init__(self):
        super().__init__('cuda')
        self.topology_simple_dual_loop()
        # self.topology_lookahead_goal()

    def topology_lookahead_goal(self):
        dataset_node = DatasetSequenceMNISTNode(
            DatasetMNISTParams(one_hot_labels=False, examples_per_class=1),
            DatasetSequenceMNISTNodeParams(seqs=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                                           transition_probs=np.array([
                                               [0, 0.5, 0.5],
                                               [1, 0, 0],
                                               [1, 0, 0]
                                           ]))
        )
        unsqueeze_node_child = UnsqueezeNode(0)
        unsqueeze_node_parent = UnsqueezeNode(0)

        expert_node_child = ExpertFlockNode(
            ExpertParams(flock_size=1,
                         n_cluster_centers=9,
                         spatial=SpatialPoolerParams(
                         ),
                         temporal=TemporalPoolerParams(
                             n_frequent_seqs=20,
                             seq_length=3,
                             seq_lookahead=1

                         )
                         )
        )
        expert_node_parent = ExpertFlockNode(
            ExpertParams(flock_size=1,
                         n_cluster_centers=3,
                         spatial=SpatialPoolerParams(
                         ))
        )

        self.add_node(dataset_node)
        self.add_node(unsqueeze_node_child)
        self.add_node(unsqueeze_node_parent)
        self.add_node(expert_node_child)
        self.add_node(expert_node_parent)

        Connector.connect(dataset_node.outputs.data, unsqueeze_node_child.inputs.input)
        Connector.connect(dataset_node.outputs.sequence_id, unsqueeze_node_parent.inputs.input)
        Connector.connect(unsqueeze_node_child.outputs.output, expert_node_child.inputs.sp.data_input)
        Connector.connect(unsqueeze_node_parent.outputs.output, expert_node_parent.inputs.sp.data_input)

        Connector.connect(expert_node_parent.outputs.output_context, expert_node_child.inputs.tp.context_input,
                          is_backward=True)

    def topology_simple_dual_loop(self):
        dataset_node = DatasetSequenceMNISTNode(
            DatasetMNISTParams(one_hot_labels=False, examples_per_class=1),
            DatasetSequenceMNISTNodeParams(seqs=[[0, 1, 2], [0, 1, 3]])
        )
        flock_size = 1

        unsqueeze_node_child = UnsqueezeNode(0)
        unsqueeze_node_parent = UnsqueezeNode(0)
        expand_node_child = ExpandNode(0, flock_size)
        expand_node_parent = ExpandNode(0, flock_size)

        expert_node_child = ExpertFlockNode(
            ExpertParams(flock_size=flock_size,
                         n_cluster_centers=4,
                         spatial=SpatialPoolerParams(
                         ),
                         temporal=TemporalPoolerParams(
                             incoming_context_size=2,
                             n_providers=2,
                             n_frequent_seqs=8,
                             seq_length=3,
                             seq_lookahead=1

                         )
                         )
        )
        expert_node_parent = ExpertFlockNode(
            ExpertParams(flock_size=flock_size,
                         n_cluster_centers=2,
                         spatial=SpatialPoolerParams(
                         ))
        )

        self.add_node(dataset_node)
        self.add_node(unsqueeze_node_child)
        self.add_node(unsqueeze_node_parent)
        self.add_node(expand_node_child)
        self.add_node(expand_node_parent)
        self.add_node(expert_node_child)
        self.add_node(expert_node_parent)

        Connector.connect(dataset_node.outputs.data, unsqueeze_node_child.inputs.input)
        Connector.connect(dataset_node.outputs.sequence_id, unsqueeze_node_parent.inputs.input)
        Connector.connect(unsqueeze_node_child.outputs.output, expand_node_child.inputs.input)
        Connector.connect(unsqueeze_node_parent.outputs.output, expand_node_parent.inputs.input)
        Connector.connect(expand_node_child.outputs.output, expert_node_child.inputs.sp.data_input)
        Connector.connect(expand_node_parent.outputs.output, expert_node_parent.inputs.sp.data_input)

        Connector.connect(expert_node_parent.outputs.output_context, expert_node_child.inputs.tp.context_input,
                          is_backward=True)
