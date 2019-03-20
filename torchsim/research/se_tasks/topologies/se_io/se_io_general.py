from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorNode
from torchsim.research.se_tasks.topologies.se_io.se_io_base import SeIoBase
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig


class SeIoGeneral(SeIoBase):
    """SpaceEngineers for task 0 (without blank actions etc..)."""

    def is_in_training_phase(self) -> bool:
        return not bool(self.get_testing_phase())

    def get_image_width(self):
        return self.se_config.render_width

    def get_image_height(self):
        return self.se_config.render_height

    actions_descriptor: SpaceEngineersActionsDescriptor
    _node_se_connector: SpaceEngineersConnectorNode

    _curriculum: tuple

    def __init__(self, curriculum: tuple = (0, -1)):
        self._curriculum = curriculum
        self.se_config = SpaceEngineersConnectorConfig()
        self.se_config.curriculum = list(self._curriculum)

    def _create_and_add_nodes(self):
        self.actions_descriptor = SpaceEngineersActionsDescriptor()
        self._node_se_connector = SpaceEngineersConnectorNode(self.actions_descriptor, self.se_config)

        # common IO
        self.outputs = self._node_se_connector.outputs
        self.inputs = self._node_se_connector.inputs

    def _add_nodes(self, target_group: NodeGroupBase):
        for node in [self._node_se_connector]:
            target_group.add_node(node)

    def _connect_nodes(self):
        pass

    def get_num_labels(self):
        return self.se_config.task_to_agent_buffer_size

    def get_image_numel(self):
        return self.se_config.render_height * self.se_config.render_width * 3

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