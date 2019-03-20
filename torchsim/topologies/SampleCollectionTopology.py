from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.nodes.actions_monitor_node import ActionMonitorNode
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorNode
from torchsim.core.nodes.constant_node import ConstantNode
from torchsim.utils.sample_collection_overseer import SampleCollectionOverseer
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig


class SEDatasetSampleCollectionTopology(Topology):
    """A model which receives data from the SE dataset and learns spatial representation from this."""

    TASK: int = 0
    SIZE: SeDatasetSize = SeDatasetSize.SIZE_64

    _node_se_connector: SpaceEngineersConnectorNode
    _node_action_monitor: ActionMonitorNode

    def __init__(self):
        super().__init__(device='cuda')
        self._actions_descriptor = SpaceEngineersActionsDescriptor()
        self._se_config = SpaceEngineersConnectorConfig()
        self._se_config.skip_frames = 1
        self._se_config.render_width = self.SIZE.value
        self._se_config.render_height = self.SIZE.value
        num_training_trajectories = 3000
        num_testing_trajectories = 100

        overseer = SampleCollectionOverseer(self._se_config.render_width, self._se_config.render_height,
                                            num_training_trajectories, num_testing_trajectories)

        self._node_se_connector = SpaceEngineersConnectorNode(self._actions_descriptor, self._se_config,
                                                              sample_collection_overseer=overseer)
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
