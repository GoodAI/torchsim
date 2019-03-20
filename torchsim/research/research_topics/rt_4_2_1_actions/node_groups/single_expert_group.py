from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ExpertFlockNode
from torchsim.research.research_topics.rt_4_2_1_actions.topologies.goal_directed_template_topology import \
    GoalDirectedExpertGroupBase


class SingleExpertGroup(GoalDirectedExpertGroupBase):
    def __init__(self):
        super().__init__("Single Expert")

        expert_params = ExpertParams()

        expert_params.flock_size = 1
        expert_params.n_cluster_centers = 24
        expert_params.produce_actions = True
        expert_params.temporal.seq_length = 9
        expert_params.temporal.seq_lookahead = 7
        expert_params.temporal.n_frequent_seqs = 700
        expert_params.temporal.max_encountered_seqs = 1000
        expert_params.temporal.exploration_probability = 0.01
        expert_params.temporal.batch_size = 200
        expert_params.temporal.own_rewards_weight = 20
        expert_params.temporal.compute_backward_pass = True

        expert_params.compute_reconstruction = True

        expert_node = ExpertFlockNode(expert_params)

        self.add_node(expert_node)

        Connector.connect(self.inputs.data.output, expert_node.inputs.sp.data_input)
        Connector.connect(self.inputs.reward.output, expert_node.inputs.tp.reward_input)

        Connector.connect(expert_node.outputs.sp.predicted_reconstructed_input,
                          self.outputs.predicted_reconstructed_input.input)
