from typing import Iterable, List

from torchsim.core.graph.node_base import NodeBase

_node_processed = -2
_node_not_visited = -1


class IllegalCycleException(Exception):
    def __init__(self, nodes: List[NodeBase]):
        ids = [node.id for node in nodes]
        self.message = f"Illegal cycle with no backward edges detected in {ids}"


def order_nodes(nodes: Iterable[NodeBase]):
    """Calculates topological ordering of the nodes.

    The first node has an order of 1.
    """
    # We don't know where the graph starts, or what parts are connected, so we have to check for cycles from all nodes
    for node in nodes:
        _detect_illegal_cycles(node, [])

    _clear_ordering(nodes)
    destinations = _find_destinations(nodes)

    ordered_nodes = []

    for destination in destinations:
        _visit_node(destination, nodes, ordered_nodes)

    for order, node in enumerate(ordered_nodes):
        node.topological_order = order + 1

    return ordered_nodes


def _clear_ordering(nodes: Iterable[NodeBase]):
    for node in nodes:
        node.topological_order = _node_not_visited


def _visit_node(current_node: NodeBase, nodes: Iterable[NodeBase], ordered_nodes: List[NodeBase]):
    if current_node.topological_order != _node_not_visited:
        # The node was already visited.
        return

    # Mark node as processed.
    current_node.topological_order = _node_processed

    for input_slot in current_node.inputs:
        connection = input_slot.connection
        if connection is None:
            continue

        if connection.is_backward:
            continue

        source_node = connection.source.owner
        if source_node in nodes:
            _visit_node(source_node, nodes, ordered_nodes)

    ordered_nodes.append(current_node)


def _find_destinations(nodes: Iterable[NodeBase]):
    destinations = list(nodes)

    for node in nodes:
        for input_slot in node.inputs:
            connection = input_slot.connection
            if connection is None or connection.is_backward:
                continue

            source_node = connection.source.owner
            if source_node in nodes and source_node in destinations:
                destinations.remove(source_node)

    return destinations


def _detect_illegal_cycles(node: NodeBase, visited_nodes: List[NodeBase], low_priority_edge_found: bool = False):
    """Recursively explores all outgoing connections from a node looking for cycles.

    Recurses until there is no outgoing connections to explore, or until it visits a node which is has done before.
    On encountering a previously visited node, it will throw an exception none of the connections were low priority.

    WARN: This has to be initially run from each node in the topology if the topological ordering is not known.
    """
    if node in visited_nodes:
        if low_priority_edge_found:
            return
        else:
            raise IllegalCycleException(visited_nodes)

    visited_nodes.append(node)
    for block in node.outputs:
        for connection in block.connections:
            _detect_illegal_cycles(connection.target.owner, list(visited_nodes),
                                   low_priority_edge_found or connection.is_backward)
