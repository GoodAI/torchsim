from typing import List

from torchsim.core.actions import AgentActionsDescriptor
from torchsim.core.nodes.scatter_node import ScatterNode


class AgentActionsParserNode(ScatterNode):
    """Scatters input values to output according to input_actions and action_descriptor.

    Input actions can be subset of all actions or have different order.
    Unset values are set to zero.

    Example:
        action_descriptor: ['right', 'left', 'up', down']
        input_actions: ['up', 'left']

        input = [1, .5]
        output = [0, .5, 1, 0]
    """

    def __init__(self, action_descriptor: AgentActionsDescriptor, input_actions: List[str], name="ActionParser",
                 device="cuda"):
        names = action_descriptor.action_names()
        mapping = [names.index(name) for name in input_actions]
        super().__init__(mapping=mapping, output_shape=(action_descriptor.ACTION_COUNT,), device=device, name=name)
