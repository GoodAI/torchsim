from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ExpertFlockNode
from torchsim.research.research_topics.rt_4_2_1_actions.topologies.goal_directed_template_topology import \
    GoalDirectedExpertGroupBase


class TwoExpertsGroup(GoalDirectedExpertGroupBase):
    def __init__(self, c_n_ccs, c_buffer_size, c_seq_length, c_seq_lookahead, p_seq_length, p_seq_lookahead, p_n_ccs,
                 flock_size):
        super().__init__("Two Experts")

        expert_params1 = ExpertParams()

        expert_params1.flock_size = flock_size
        expert_params1.n_cluster_centers = c_n_ccs
        expert_params1.produce_actions = True
        expert_params1.temporal.seq_length = c_seq_length
        expert_params1.temporal.seq_lookahead = c_seq_lookahead
        expert_params1.temporal.n_frequent_seqs = 700
        expert_params1.temporal.max_encountered_seqs = 1000
        expert_params1.temporal.exploration_probability = 0.05
        expert_params1.temporal.batch_size = 200
        expert_params1.temporal.compute_backward_pass = True
        expert_params1.temporal.frustration_threshold = 2

        expert_params2 = expert_params1.clone()

        expert_params1.spatial.buffer_size = c_buffer_size
        expert_params1.compute_reconstruction = True

        expert_params2.temporal.seq_length = p_seq_length
        expert_params2.temporal.seq_lookahead = p_seq_lookahead
        expert_params2.n_cluster_centers = p_n_ccs
        expert_params2.produce_actions = False
        expert_params2.temporal.frustration_threshold = 10

        expert_node1 = ExpertFlockNode(expert_params1)
        expert_node2 = ExpertFlockNode(expert_params2)

        self.add_node(expert_node1)
        self.add_node(expert_node2)

        Connector.connect(self.inputs.data.output, expert_node1.inputs.sp.data_input)
        Connector.connect(self.inputs.reward.output, expert_node1.inputs.tp.reward_input)
        Connector.connect(self.inputs.reward.output, expert_node2.inputs.tp.reward_input)

        # Connect the experts to each other.
        Connector.connect(expert_node1.outputs.tp.projection_outputs, expert_node2.inputs.sp.data_input)
        Connector.connect(expert_node2.outputs.output_context, expert_node1.inputs.tp.context_input, is_backward=True)

        # Connect the group output.
        Connector.connect(expert_node1.outputs.sp.predicted_reconstructed_input,
                          self.outputs.predicted_reconstructed_input.input)
