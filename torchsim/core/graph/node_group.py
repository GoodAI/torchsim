from abc import ABC
from typing import List, Generic

from torchsim.core.exceptions import IllegalStateException
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.id_generator import IdGenerator
from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket, InversePassInputPacket
from torchsim.core.graph.invertible_node import InvertibleNode
from torchsim.core.graph.node_base import NodeBase, TInputs, TInternals, TOutputs, EmptyOutputs
from torchsim.core.graph.node_ordering import order_nodes
from torchsim.core.graph.slot_container_base import InputsBase, OutputsBase, TNodeBase
from torchsim.core.graph.slots import GroupVirtualOutputSlot, GroupInputSlot, GroupOutputSlot
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.gui.observables import ObserverPropertiesItem, ObserverPropertiesBuilder


class GenericGroupInputs(InputsBase[GroupInputSlot, TNodeBase], Generic[TNodeBase]):
    """A basic class for group inputs.

    Subclass this to specify inputs for a specialized node group.
    """

    def _create_slot_instance(self, name) -> GroupInputSlot:
        return GroupInputSlot(self._owner, name)


class GroupInputs(GenericGroupInputs[NodeBase]):
    pass


class SimpleGroupInputs(GroupInputs):
    """A simple GroupInputs implementation which creates a given number of generic group inputs."""

    def __init__(self, owner: 'NodeGroupBase', n_inputs: int):
        super().__init__(owner)
        for i in range(n_inputs):
            self.create(f"Input {i}")


class GenericGroupOutputs(OutputsBase[GroupOutputSlot, TNodeBase]):
    """A basic class for group outputs.

    Subclass this to specify outputs for a specialized node group.
    """

    def _create_slot_instance(self, name) -> GroupOutputSlot:
        return GroupOutputSlot(self._owner, name)

    def connect_automatically(self, group_inputs: GroupInputs, is_backward: bool = False):
        """Automatically connect outputs from this container with inputs from group_inputs.

        The slots are matched by name.
        """

        input_slots = {slot.name: slot for slot in group_inputs._slots}

        for output_slot in self._slots:
            input_slot = input_slots.get(output_slot.name, None)
            if input_slot is None:
                continue

            Connector.connect(output_slot, input_slot, is_backward)


class GroupOutputs(GenericGroupOutputs[NodeBase]):
    pass


class SimpleGroupOutputs(GroupOutputs):
    """A simple GroupOutputs implementation which creates a given number of generic group outputs."""

    def __init__(self, owner: 'NodeGroupBase', n_outputs: int):
        super().__init__(owner)
        for i in range(n_outputs):
            self.create(f"Output {i}")


class NodeAlreadyPresentException(Exception):
    pass


class NodeGroupWithInternalsBase(NodeBase[TInputs, TInternals, TOutputs], InvertibleNode, ABC):
    """A base class for a node group which has internal slots.

    A node group is viewed as a node from the outside. Internally, all the group inputs will have a member called
    `output`, to which the internal nodes can connect, and which will in turn provide the tensor connected to
    the associated node's input. All the group's outputs have an `input` member, to which can the internal nodes'
    outputs be connected.

    The nodes inside the group have their own ordering and are executed when the group's step() method is called.

    For a group which does not have internals, see NodeGroupBase.

    Usage:
    class CustomGroup(NodeGroupWithInternalsBase[CustomInputs, CustomInternals, CustomOutputs]):
        ...

    Note: We might actually remove this if we find out that no groups need to use this.
    """
    _is_initialized: bool

    nodes: List[NodeBase]
    _ordered_nodes: List[NodeBase]

    _prop_builder: ObserverPropertiesBuilder

    def __init__(self, name: str, inputs: TInputs = None, internals: TInternals = None,
                 outputs: TOutputs = None):
        """Initializes the NodeGroup.

        Args:
            name: The name of the group.
            inputs: See NodeBase.
            internals: See NodeBase.
            outputs: See NodeBase.
        """
        super().__init__(name, inputs=inputs, memory_blocks=internals, outputs=outputs)
        self.nodes = []
        self._ordered_nodes = []
        self._is_initialized = False

    def is_initialized(self):
        # return all([n.is_initialized() for n in self._ordered_nodes])
        return self._is_initialized

    def add_node(self, node: TNodeBase) -> TNodeBase:
        if node in self.nodes:
            raise NodeAlreadyPresentException()

        if self.is_initialized():
            # When adding nodes to initialized topology, it must be ensured that: ids are assigned, nodes are ordered,
            # tensor sizes negotiation finished, etc.
            # This has to be implemented, so it's not safe now to add nodes
            raise IllegalStateException("Cannot add node to initialized topology. Stop it first or implement this functionality.")
        self.nodes.append(node)

        return node

    def create_generic_node_group(self, name, n_inputs, n_outputs) -> 'GenericNodeGroup':
        group = GenericNodeGroup(name, n_inputs, n_outputs)
        self.add_node(group)

        return group

    def add_nodes(self, nodes_list: List[NodeBase]):
        for node in nodes_list:
            self.add_node(node)

    def remove_node(self, node: NodeBase):
        self.nodes.remove(node)

    def _assign_ids_to_nodes(self, id_generator: IdGenerator):
        for node in self.nodes:
            # Assign ids to nodes which don't have them yet.
            if node.id == 0:
                node.id = id_generator.next_node_id()

            if isinstance(node, NodeGroupBase):
                node._assign_ids_to_nodes(id_generator)

    def order_nodes(self):
        self._ordered_nodes = order_nodes(self.nodes)

        for node in self._ordered_nodes:
            if isinstance(node, NodeGroupBase):
                node.order_nodes()

    def _step(self):
        for node in self._ordered_nodes:
            node.step()

    def allocate_memory_blocks(self, tensor_creator: TensorCreator):
        for node in self._ordered_nodes:
            node.allocate_memory_blocks(tensor_creator)
        self._is_initialized = True

    def release_memory_blocks(self):
        for node in self._ordered_nodes:
            node.release_memory_blocks()

        self._is_initialized = False

    def detect_dims_change(self) -> bool:
        children_sizes_changed = False
        for node in self._ordered_nodes:
            children_sizes_changed |= node.detect_dims_change()

        return children_sizes_changed

    def validate(self):
        for node in self._ordered_nodes:
            node.validate()

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        slot: GroupVirtualOutputSlot = data.slot
        true_slot = slot.input
        return [InversePassInputPacket(data.tensor, true_slot)]

    def get_properties(self) -> List[ObserverPropertiesItem]:
        prop_list = []
        for node in self.nodes:
            node_properties = node.get_properties()
            if len(node_properties) > 0:
                prop_list.append(self._prop_builder.collapsible_header(f'{node.name_with_id} - Properties', False))
                prop_list.extend([ObserverPropertiesItem.clone_with_prefix(n, f'#{node.id} ') for n in node_properties])

        return prop_list

    def _save(self, saver: Saver):
        for node in self.nodes:
            node.save(saver)

    def _load(self, loader: Loader):
        for node in self.nodes:
            node.load(loader)


class NodeGroupBase(NodeGroupWithInternalsBase[TInputs, EmptyOutputs, TOutputs], ABC):
    """A base for node groups which do not have internal slots."""
    pass


class GenericNodeGroup(NodeGroupBase[SimpleGroupInputs, SimpleGroupOutputs]):
    """A generic node group which only has a number of annonymous inputs and outputs."""
    inputs: SimpleGroupInputs
    outputs: SimpleGroupOutputs

    def __init__(self, name: str, n_inputs: int, n_outputs: int):
        super().__init__(name, inputs=SimpleGroupInputs(self, n_inputs), outputs=SimpleGroupOutputs(self, n_outputs))
