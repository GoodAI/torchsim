from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ActionMonitorNode
from torchsim.core.nodes import ExpertFlockNode
from torchsim.core.nodes import ReceptiveFieldNode
from torchsim.core.nodes import SpaceEngineersConnectorNode
from torchsim.core.nodes import ConstantNode
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.utils.param_utils import Size2D
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig

EOX = 2  # no experts on X (SX has to be divisible by this)
EOY = 2  # no experts on Y (SY has to be divisible by this)


class SeToyArchDebugTopology(Topology):
    """Basic test of the SE and TA interaction."""

    _node_se_connector: SpaceEngineersConnectorNode
    _node_action_monitor: ActionMonitorNode
    _node_flock: ExpertFlockNode

    def get_expert_params(self):
        expert_params = ExpertParams()
        expert_params.flock_size = EOX * EOY
        expert_params.n_cluster_centers = 30
        expert_params.spatial.buffer_size = 500
        expert_params.temporal.buffer_size = 100
        expert_params.temporal.incoming_context_size = 10
        expert_params.spatial.batch_size = 50
        expert_params.temporal.batch_size = 50
        expert_params.spatial.input_size = 3 * self.SX // EOX * self.SY // EOY
        return expert_params

    def __init__(self):
        super().__init__(device='cuda')

        self._se_config = SpaceEngineersConnectorConfig()
        self.SX = self._se_config.render_width
        self.SY = self._se_config.render_height

        # setup the params
        # SeToyArchDebugTopology.config_se_communication()
        expert_params = self.get_expert_params()

        # create the nodes
        self._actions_descriptor = SpaceEngineersActionsDescriptor()
        self._node_se_connector = SpaceEngineersConnectorNode(self._actions_descriptor, self._se_config)
        self._node_action_monitor = ActionMonitorNode(self._actions_descriptor)

        self._blank_action = ConstantNode(shape=self._actions_descriptor.ACTION_COUNT, constant=0)
        self._blank_task_data = ConstantNode(shape=self._se_config.agent_to_task_buffer_size, constant=0)
        self._blank_task_control = ConstantNode(shape=self._se_config.TASK_CONTROL_SIZE, constant=0)

        # parent_rf_dims = (self.lrf_width, self.lrf_height, 3)
        self._node_lrf = ReceptiveFieldNode((self.SY, self.SX, 3), Size2D(self.SX // EOX, self.SY // EOY))

        self._node_flock = ExpertFlockNode(expert_params)

        self.add_node(self._node_se_connector)
        self.add_node(self._node_action_monitor)
        self.add_node(self._blank_action)
        self.add_node(self._blank_task_data)
        self.add_node(self._blank_task_control)
        self.add_node(self._node_flock)
        self.add_node(self._node_lrf)

        Connector.connect(self._blank_action.outputs.output,
                          self._node_action_monitor.inputs.action_in)
        Connector.connect(self._node_action_monitor.outputs.action_out,
                          self._node_se_connector.inputs.agent_action)

        Connector.connect(self._blank_task_data.outputs.output,
                          self._node_se_connector.inputs.agent_to_task_label)
        Connector.connect(self._blank_task_control.outputs.output,
                          self._node_se_connector.inputs.task_control)

        # SE -> Flock (no LRF)
        # Connector.connect(self._node_se_connector.outputs.image_output,
        # self._node_flock.inputs.sp.data_input)

        # Image -> LRF -> Flock
        Connector.connect(self._node_se_connector.outputs.image_output,
                          self._node_lrf.inputs[0])
        Connector.connect(self._node_lrf.outputs[0],
                          self._node_flock.inputs.sp.data_input)
