from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase, GroupInputs, GroupOutputs
from torchsim.core.nodes.actions_monitor_node import ActionMonitorNode
from torchsim.core.nodes.agent_actions_parser_node import AgentActionsParserNode
from torchsim.core.nodes.pass_node import PassNode
from torchsim.core.nodes.random_number_node import RandomNumberNode
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorNode
from torchsim.core.nodes.switch_node import SwitchNode
from torchsim.core.nodes.to_one_hot_node import ToOneHotNode
from torchsim.core.nodes.constant_node import ConstantNode
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig


class Task1BaseGroupWorldInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.last_actions = self.create("Last actions")
        self.current_actions = self.create("Current actions")


class Task1BaseGroupWorldOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.actions = self.create("Actions")
        self.image = self.create("Image")
        self.current_location = self.create("Current location")
        self.target_location = self.create("Target location")


class Task1BaseGroupWorld(NodeGroupBase[Task1BaseGroupWorldInputs, Task1BaseGroupWorldOutputs]):
    def __init__(self, curriculum: tuple = (1, -1)):
        super().__init__("Task 1 - Base topology world", inputs=Task1BaseGroupWorldInputs(self),
                         outputs=Task1BaseGroupWorldOutputs(self))

        actions_descriptor = SpaceEngineersActionsDescriptor()

        actions = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']
        action_count = len(actions)

        # SE nodes
        se_config = SpaceEngineersConnectorConfig()
        se_config.curriculum = list(curriculum)

        node_se_connector = SpaceEngineersConnectorNode(actions_descriptor, se_config)

        def node_se_connector_is_learning() -> bool:
            if node_se_connector.outputs.metadata_testing_phase.tensor is None:
                return False
            else:
                return node_se_connector.outputs.metadata_testing_phase.tensor.cpu().item() == 1

        node_se_connector.is_learning = node_se_connector_is_learning

        self.node_se_connector = node_se_connector

        blank_task_control = ConstantNode((se_config.TASK_CONTROL_SIZE,))

        pass_actions_node = PassNode(output_shape=(action_count,), name="pass actions")

        blank_task_labels = ConstantNode((20,))

        self.add_node(node_se_connector)
        self.add_node(blank_task_control)
        self.add_node(pass_actions_node)
        self.add_node(blank_task_labels)

        Connector.connect(
            self.inputs.last_actions.output,
            pass_actions_node.inputs.input
        )

        Connector.connect(
            self.inputs.current_actions.output,
            node_se_connector.inputs.agent_action
        )

        Connector.connect(
            pass_actions_node.outputs.output,
            self.outputs.actions.input
        )

        Connector.connect(
            node_se_connector.outputs.image_output,
            self.outputs.image.input
        )

        Connector.connect(
            node_se_connector.outputs.task_to_agent_location_one_hot,
            self.outputs.current_location.input
        )

        Connector.connect(
            node_se_connector.outputs.task_to_agent_location_target_one_hot,
            self.outputs.target_location.input
        )

        # blank connection
        Connector.connect(blank_task_control.outputs.output,
                          node_se_connector.inputs.task_control)
        Connector.connect(blank_task_labels.outputs.output,
                          node_se_connector.inputs.agent_to_task_label)


class Task1BaseGroupInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.actions = self.create("Actions")


class Task1BaseGroupOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.last_actions = self.create("Last actions")
        self.current_actions = self.create("Current actions")


class Task1BaseGroup(NodeGroupBase[Task1BaseGroupInputs, Task1BaseGroupOutputs]):
    RANDOM_INPUT_ID: int = 0
    ARCHITECTURE_INPUT_ID: int = 1

    def __init__(self):
        super().__init__("Task 1 - Base topology", inputs=Task1BaseGroupInputs(self),
                         outputs=Task1BaseGroupOutputs(self))

        actions_descriptor = SpaceEngineersActionsDescriptor()

        actions = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']
        action_count = len(actions)

        to_one_hot_node = ToOneHotNode()

        node_action_monitor = ActionMonitorNode(actions_descriptor)
        action_parser_node = AgentActionsParserNode(actions_descriptor, actions)

        random_node = RandomNumberNode(0, action_count, name="random action generator", generate_new_every_n=5,
                                       randomize_intervals=True)

        switch_node = SwitchNode(2, active_input_index=self.RANDOM_INPUT_ID)

        def switch_node_switch_learning(on):
            if on:
                switch_node.change_input(self.RANDOM_INPUT_ID)
            else:
                switch_node.change_input(self.ARCHITECTURE_INPUT_ID)

        switch_node.switch_learning = switch_node_switch_learning

        self.add_node(node_action_monitor)
        self.add_node(to_one_hot_node)
        self.add_node(action_parser_node)
        self.add_node(random_node)
        self.add_node(switch_node)

        Connector.connect(
            self.inputs.actions.output,
            to_one_hot_node.inputs.input
        )

        Connector.connect(
            random_node.outputs.one_hot_output,
            switch_node.inputs[0]
        )
        Connector.connect(
            to_one_hot_node.outputs.output,
            switch_node.inputs[1]
        )
        Connector.connect(
            switch_node.outputs.output,
            action_parser_node.inputs.input
        )
        Connector.connect(
            action_parser_node.outputs.output,
            node_action_monitor.inputs.action_in
        )
        Connector.connect(
            switch_node.outputs.output,
            self.outputs.last_actions.input
        )
        Connector.connect(
            node_action_monitor.outputs.action_out,
            self.outputs.current_actions.input
        )
