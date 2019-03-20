from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.nodes.internals.grid_world import GridWorldActionDescriptor, GridWorldParams
from torchsim.core.nodes import ActionMonitorNode
from torchsim.core.nodes import GridWorldNode
from torchsim.core.nodes import RandomNumberNode


class GridWorldTopology(Topology):
    _node_grid_world: GridWorldNode
    _node_action_monitor: ActionMonitorNode

    def __init__(self):
        super().__init__(device='cuda')
        actions_descriptor = GridWorldActionDescriptor()
        node_action_monitor = ActionMonitorNode(actions_descriptor)

        params = GridWorldParams()
        node_grid_world = GridWorldNode(params)

        random_action_generator = RandomNumberNode(upper_bound=len(actions_descriptor.action_names()))

        self.add_node(node_grid_world)
        self.add_node(node_action_monitor)
        self.add_node(random_action_generator)

        Connector.connect(random_action_generator.outputs.one_hot_output,
                          node_action_monitor.inputs.action_in)
        Connector.connect(node_action_monitor.outputs.action_out,
                          node_grid_world.inputs.agent_action)
