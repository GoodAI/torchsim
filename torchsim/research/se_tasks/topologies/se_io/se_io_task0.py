from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes.actions_monitor_node import ActionMonitorNode
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorNode
from torchsim.core.nodes.constant_node import ConstantNode
from torchsim.research.se_tasks.topologies.se_io.se_io_general import SeIoGeneral


class SeIoTask0(SeIoGeneral):
    """SpaceEngineers for task 0 (blank actions etc..)."""

    actions_descriptor: SpaceEngineersActionsDescriptor
    _node_se_connector: SpaceEngineersConnectorNode
    _node_action_monitor: ActionMonitorNode

    _blank_action: ConstantNode
    _blank_task_data = ConstantNode
    _blank_task_control = ConstantNode

    def __init__(self, curriculum: tuple = (0, -1)):
        super().__init__(curriculum)

    def _create_and_add_nodes(self):
        super()._create_and_add_nodes()

        self._node_action_monitor = ActionMonitorNode(self.actions_descriptor)
        self._blank_action = ConstantNode(self.actions_descriptor.ACTION_COUNT)
        self._blank_task_data = ConstantNode(self.get_num_labels())
        self._blank_task_control = ConstantNode(self.se_config.TASK_CONTROL_SIZE)

    def _add_nodes(self, target_group: NodeGroupBase):
        super()._add_nodes(target_group)

        for node in [self._node_action_monitor,
                     self._blank_action,
                     self._blank_task_data,
                     self._blank_task_control]:
            target_group.add_node(node)

    def _connect_nodes(self):
        super()._connect_nodes()

        Connector.connect(
            self._blank_action.outputs.output,
            self._node_action_monitor.inputs.action_in)
        Connector.connect(
            self._node_action_monitor.outputs.action_out,
            self._node_se_connector.inputs.agent_action)
        Connector.connect(
            self._blank_task_control.outputs.output,
            self._node_se_connector.inputs.task_control)


    def get_image_numel(self):
        return self.se_config.render_height * self.se_config.render_width * 3

    def get_image_width(self):
        return self.se_config.render_width

    def get_image_height(self):
        return self.se_config.render_height

    def get_task_id(self) -> float:
        return self._node_se_connector.outputs.metadata_task_id.tensor.cpu().item()

    def get_task_instance_id(self) -> float:
        return self._node_se_connector.outputs.metadata_task_instance_id.tensor.cpu().item()

    def get_task_status(self) -> float:
        return self._node_se_connector.outputs.metadata_task_status.tensor.cpu().item()

    def get_task_instance_status(self) -> float:
        return self._node_se_connector.outputs.metadata_task_instance_status.tensor.cpu().item()

    def get_reward(self) -> float:
        return self._node_se_connector.outputs.reward_output.tensor.cpu().item()

    def get_testing_phase(self) -> float:
        if self._node_se_connector.outputs.metadata_testing_phase.tensor is None:
            return False
        else:
            return self._node_se_connector.outputs.metadata_testing_phase.tensor.cpu().item()
