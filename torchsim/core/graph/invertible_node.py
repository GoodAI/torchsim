from abc import ABC, abstractmethod
from typing import List

from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket, InversePassInputPacket


class InvertibleNode(ABC):
    """A mixin adding support the inverse projection (used with NodeBase/WorkNodeBase)."""
    @abstractmethod
    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        """Compute interpretation of data which is associated with an output memory block in the input space(s).

        Note that the inverse pass packets must not have the input memory block associated with them, but the source
        block from the connection.

        Args:
            data: The packed associated with a memory block of this node.

        Returns:
            A list of packets associated with the memory block of nodes whence the inputs came.
        """
        pass

    def recursive_inverse_projection_from_output(self, data: InversePassOutputPacket) -> List[InversePassOutputPacket]:
        """Compute the inverse projection in the graph from a packet associated this node's output."""
        terminals = []
        self._recursive_inverse_projection_from_output(data, terminals)
        return terminals

    def recursive_inverse_projection_from_input(self, data: InversePassInputPacket) -> List[InversePassOutputPacket]:
        """Compute the inverse projection in the graph from a packet associated this node's input."""
        terminals = []
        self._recursive_inverse_projection_from_input(data, terminals)
        return terminals

    @staticmethod
    def _recursive_inverse_projection_from_input(data: InversePassInputPacket,
                                                 terminals: List[InversePassOutputPacket]):

        output_packet = InversePassOutputPacket(data.tensor, data.slot.source)

        if isinstance(output_packet.slot.owner, InvertibleNode) and not data.slot.connection.is_backward:
            owner: InvertibleNode = output_packet.slot.owner
            owner._recursive_inverse_projection_from_output(output_packet, terminals)
        else:
            for terminal in terminals:
                if terminal.slot == output_packet.slot:
                    # Integrate the tensor.
                    terminal.tensor += output_packet.tensor
                    break
            else:
                terminals.append(output_packet)

    def _recursive_inverse_projection_from_output(self, data: InversePassOutputPacket,
                                                  terminals: List[InversePassOutputPacket]):
        """Recursively go backwards through the network of Nodes, make inverse projections and collect the results.

        If a terminal memory block (non-invertible) is found, a reconstruction packet is created and added to the list
        of terminals. If a packet with the same memory block is found in terminals, it is used instead.

        The resulting reconstructions are in stored in terminals.
        """
        # Project data into the input space packets.
        inverse_input_packets = self._inverse_projection(data)

        for input_packet in inverse_input_packets:
            self._recursive_inverse_projection_from_input(input_packet, terminals)
