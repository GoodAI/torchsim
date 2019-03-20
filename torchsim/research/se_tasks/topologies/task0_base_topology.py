from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.graph.node_group import NodeGroupBase, GroupInputs, GroupOutputs
from torchsim.core.graph.slots import GroupInputSlot, GroupOutputSlot
from torchsim.core.nodes.dataset_se_objects_node import DatasetConfig, DatasetSeObjectsParams, DatasetSeObjectsNode
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorNode
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig


class Task0BaseGroupInputs(GroupInputs):
    labels: GroupInputSlot

    def __init__(self, owner):
        super().__init__(owner)
        self.labels = self.create("Labels")


class Task0BaseGroupOutputs(GroupOutputs):
    image: GroupOutputSlot
    labels: GroupOutputSlot

    def __init__(self, owner):
        super().__init__(owner)
        self.image = self.create("Image")
        self.labels = self.create("Labels")


class Task0BaseGroup(NodeGroupBase[Task0BaseGroupInputs, Task0BaseGroupOutputs]):
    def __init__(self, use_dataset: bool = True):
        super().__init__("Task 0 - Base topology", inputs=Task0BaseGroupInputs(self),
                         outputs=Task0BaseGroupOutputs(self))

        if use_dataset:
            params = DatasetSeObjectsParams(dataset_config=DatasetConfig.TRAIN_TEST, save_gpu_memory=True)
            self.se_node = DatasetSeObjectsNode(params)
        else:
            se_config = SpaceEngineersConnectorConfig()
            se_config.curriculum = list((0, -1))
            actions_descriptor = SpaceEngineersActionsDescriptor()
            self.se_node = SpaceEngineersConnectorNode(actions_descriptor, se_config)

        # self.output_image = self.se_node.outputs.image_output
        # self.output_labels = self.se_node.outputs.task_to_agent_label
        # self.input_labels = self.se_node.inputs.agent_to_task_label
