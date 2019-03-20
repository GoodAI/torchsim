from typing import Dict, List, Any

import torch

from torchsim.core.graph import Topology
from torchsim.core.graph.node_base import NodeBase
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.nodes.expert_node import ExpertFlockNode
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.core.nodes.temporal_pooler_node import TemporalPoolerFlockNode
from torchsim.core.utils.tensor_utils import same


class ComparableTensor:
    """Wraps a torch tensor and customizes the equality operator.

    We need this because
        - Comparisons between a tensor and None throw a TypeError
        - We need to call the same() function to handle NaNs
    """
    _tensor: torch.Tensor

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def __eq__(self, other: Any):
        return isinstance(other, ComparableTensor) and same(self._tensor, other._tensor)


class TAState:
    """Records the state of the toy architecture nodes.

    The state is defined by the tensors pointed to by the memory_blocks attribute.

    Currently, the state is recorded as a dictionary of nodes, attributes, and cloned tensors.
    If this is inefficient, cloned tensors could be replaced with checksums.
    """
    _state: Dict[str, Dict[str, List[ComparableTensor]]]

    def __init__(self, topology: Topology):
        self._record_state(topology)

    def _record_state(self, topology: Topology):
        """Takes a snap shot of the internal state of the topology.
        """
        self._state = {}
        for node in self.get_ta_nodes(topology):
            node_descriptor = f"{type(node)} {node.name_with_id}"
            self._state[node_descriptor] = {'memory_blocks': self._clone_block_tensors(node.memory_blocks)}

    @staticmethod
    def _clone_block_tensors(blocks: MemoryBlocks) -> List[ComparableTensor]:
        """Returns a list of ComparableTensor wrapping clones of the tensors in the memory blocks."""
        return [None if block.tensor is None else ComparableTensor(block.tensor.clone())
                for block in blocks]

    @staticmethod
    def get_ta_nodes(topology: Topology) -> List[NodeBase]:
        """Returns a list of Toy Architecture expert (and pooler) nodes."""
        return [node for node in topology.nodes if TAState._is_ta_node(node)]

    @staticmethod
    def _is_ta_node(node: NodeBase) -> bool:
        """Decides if a node is a Toy Architecture expert (or pooler) node."""
        return type(node) in [SpatialPoolerFlockNode, TemporalPoolerFlockNode, ExpertFlockNode]

    @staticmethod
    def _get_tensor_attributes(node: NodeBase) -> List[str]:
        """Returns a list of tensor-valued node attributes.

        Since nodes wrap tensors inside memory blocks, this is not currently used.
        """
        return [a for a in dir(node) if type(node.__getattribute__(a)) is torch.Tensor]

    @staticmethod
    def _get_tensor_attribute_dict(node: NodeBase) -> Dict[str, torch.Tensor]:
        """Makes a dictionary of tensor-valued node attributes.

        Since nodes wrap tensors inside memory blocks, this is not currently used.
        """
        attributes = TAState._get_tensor_attributes()
        return dict(zip(attributes, [node.__getattribute__(attribute).clone() for attribute in attributes]))

    def __eq__(self, other: "TAState") -> bool:
        """Defines equality for Toy Architecture topology states."""
        return self._state == other._state
