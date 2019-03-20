from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.nodes import ActionMonitorNode
from torchsim.core.nodes import SpaceEngineersConnectorNode
from torchsim.core.nodes import ConstantNode
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig


class DebugAgentTopology(Topology):
    _node_se_connector: SpaceEngineersConnectorNode
    _node_action_monitor: ActionMonitorNode

    def __init__(self):
        super().__init__(device='cuda')
        self._actions_descriptor = SpaceEngineersActionsDescriptor()
        self._se_config = SpaceEngineersConnectorConfig()

        self._node_se_connector = SpaceEngineersConnectorNode(self._actions_descriptor, self._se_config)
        self._node_action_monitor = ActionMonitorNode(self._actions_descriptor)

        self._blank_action = ConstantNode(shape=self._actions_descriptor.ACTION_COUNT, constant=0)
        self._blank_task_data = ConstantNode(shape=self._se_config.agent_to_task_buffer_size, constant=0)
        self._blank_task_control = ConstantNode(shape=self._se_config.TASK_CONTROL_SIZE, constant=0)

        self.add_node(self._node_se_connector)
        self.add_node(self._node_action_monitor)
        self.add_node(self._blank_action)
        self.add_node(self._blank_task_data)
        self.add_node(self._blank_task_control)

        Connector.connect(self._blank_action.outputs.output,
                          self._node_action_monitor.inputs.action_in)
        Connector.connect(self._node_action_monitor.outputs.action_out,
                          self._node_se_connector.inputs.agent_action)
        Connector.connect(self._blank_task_data.outputs.output,
                          self._node_se_connector.inputs.agent_to_task_label)
        Connector.connect(self._blank_task_control.outputs.output,
                          self._node_se_connector.inputs.task_control)
