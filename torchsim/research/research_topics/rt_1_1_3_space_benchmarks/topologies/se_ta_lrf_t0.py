
from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.nodes.actions_monitor_node import ActionMonitorNode
from torchsim.core.nodes.expert_node import ExpertFlockNode
from torchsim.core.nodes.receptive_field_node import ReceptiveFieldNode
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorNode
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import NUMBER_OF_CONTEXT_TYPES
from torchsim.core.nodes.constant_node import ConstantNode
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.benchmark_lrf_flock_topology import \
    BenchmarkLrfFlockTopology, compute_flock_sizes, setup_flock_params, init_se_dataset_world_params, compute_lrf_params
from torchsim.utils.seed_utils import set_global_seeds
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig


class SeTaLrfT0(BenchmarkLrfFlockTopology):
    """
    A model which receives data from the SE dataset and learns spatial and temporal patterns (task0)
    """

    def __init__(self,
                 seed: int=0,
                 device: str = 'cuda',
                 eox: int = 2,
                 eoy: int = 2,
                 num_cc: int = 100,
                 batch_s=300,
                 tp_learn_period=50,
                 tp_max_enc_seq=1000,
                 se_skip_frames=9):
        super().__init__(eox, eoy)

        self._se_config = SpaceEngineersConnectorConfig()
        self._se_config.skip_frames = se_skip_frames
        self._se_config.curriculum = [0, -1]

        # compute/setup parameters of the model
        _, self._sy, self._sx, self._no_channels = init_se_dataset_world_params(random_order=False)
        flock_size, input_size = compute_flock_sizes(self._sy, self._sx, self._no_channels, self._eoy, self._eox)
        expert_params = setup_flock_params(no_clusters=num_cc,
                                           buffer_size=batch_s * 2,
                                           batch_size=batch_s,
                                           tp_learn_period=tp_learn_period,
                                           max_enc_seq=tp_max_enc_seq,
                                           flock_size=flock_size,
                                           input_size=input_size)
        flock_input_size, flock_output_size = compute_lrf_params(self._sy, self._sx, self._no_channels, self._eoy,
                                                                 self._eox)

        # SE nodes
        self._actions_descriptor = SpaceEngineersActionsDescriptor()
        self._node_se_connector = SpaceEngineersConnectorNode(self._actions_descriptor, self._se_config)
        self._node_action_monitor = ActionMonitorNode(self._actions_descriptor)
        self._blank_action = ConstantNode(shape=self._actions_descriptor.ACTION_COUNT, constant=0)
        self._blank_task_data = ConstantNode(shape=self._se_config.agent_to_task_buffer_size, constant=0)

        # flock-related nodes
        self._lrf_node = ReceptiveFieldNode(flock_input_size, flock_output_size)
        self._flock_node = ExpertFlockNode(expert_params, seed=seed)
        self._zero_context = ConstantNode(shape=(expert_params.flock_size, NUMBER_OF_CONTEXT_TYPES,
                                             expert_params.temporal.incoming_context_size), constant=0)
        self._blank_task_control = ConstantNode(shape=self._se_config.TASK_CONTROL_SIZE, constant=0)

        # add nodes to the graph
        self.add_node(self._lrf_node)
        self.add_node(self._flock_node)
        self.add_node(self._zero_context)
        self.add_node(self._node_se_connector)
        self.add_node(self._node_action_monitor)
        self.add_node(self._blank_action)
        self.add_node(self._blank_task_data)
        self.add_node(self._blank_task_control)

        # connect SE -> LRF -> SP
        Connector.connect(
            self._node_se_connector.outputs.image_output,
            self._lrf_node.inputs[0])
        Connector.connect(
            self._lrf_node.outputs[0],
            self._flock_node.inputs.sp.data_input)
        Connector.connect(
            self._zero_context.outputs.output,
            self._flock_node.inputs.tp.context_input)

        # connect NOOP -> action_override
        Connector.connect(self._blank_action.outputs.output,
                          self._node_action_monitor.inputs.action_in)
        Connector.connect(self._node_action_monitor.outputs.action_out,
                          self._node_se_connector.inputs.agent_action)

        # connect blank_task_data -> SE aux input
        Connector.connect(self._blank_task_data.outputs.output,
                          self._node_se_connector.inputs.agent_to_task_label)
        Connector.connect(self._blank_task_control.outputs.output,
                          self._node_se_connector.inputs.task_control)

        # prepare for run
        set_global_seeds(seed)
        self._last_step_duration = 0


