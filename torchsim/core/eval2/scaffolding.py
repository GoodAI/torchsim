import dataclasses
import inspect
from typing import Any, Dict, Callable

from torchsim.core.eval2.experiment_template_base import TopologyFactory, TTopology
from torchsim.core.graph.node_base import NodeBase


class TopologyScaffoldingFactory(TopologyFactory[TTopology]):
    """A topology factory which builds topologies from smaller parts (nodes).

    Usage:

    class SomeGraph(TopologicalGraph):
        def __init__(self, node1, node2):
            # Connect nodes together.

    def SomeNode(NodeBase):
        def __init__(self, param=None):
            super().__init__(name="Some node")
            self.param1 = param

    factory = TopologyScaffoldingFactory(SomeGraph, node1=SomeNode, node2=SomeNode)
    params = [{'node1': {'param': 1}, 'node2': {'param': 1}},
              {'node1': {'param': 2}, 'node2': {'param': 3}}]
    """
    def __init__(self, graph_factory: Callable[..., TTopology], **plugin_factories: Callable[..., Any]):
        """Initializes the factory.

        Args:
            graph_factory: A factory method which receives the individual nodes and creates the topology.
            plugin_factories: Factory methods which create the individual nodes based on parameters. These methods must
                              take kwargs on their inputs.

        Note that the factories can also be just classes. In that case, the __init__ method is scanned for default
        parameters.

        The plugin_factories keywords must match the keywords provided to create_topology().
        """
        self._graph_factory = graph_factory
        self._plugin_factories = plugin_factories

    def create_topology(self, **kwargs) -> TTopology:
        """Creates the topology given parameters."""
        nodes = {key: self._plugin_factories[key](**kwargs[key]) for key in self._plugin_factories.keys()}
        return self._graph_factory(**nodes)

    def get_default_parameters(self) -> Dict[str, Any]:
        """Returns the default parameters for each of the plugin factories."""
        graph_factory_parameters = get_defaults(self._graph_factory).keys()
        plugin_factory_defaults = {key: get_defaults(value) for key, value in self._plugin_factories.items()}
        # Add the names of the factories (class or function name)
        for key in plugin_factory_defaults.keys():
            plugin_factory_defaults[key]['plugin_name'] = self._plugin_factories[key].__name__

        return {graph_key: plugin_factory_defaults[graph_key] for graph_key in graph_factory_parameters}


def get_defaults(f) -> Dict[str, Any]:
    """Gets the default parameters of f, where f is either a factory function, or a type.

    If f is a type, the defaults of the __init__ method will be returned.
    """
    if isinstance(f, type):
        signature = inspect.signature(f.__init__)
    else:
        signature = inspect.signature(f)

    defaults = {}
    for key, value in signature.parameters.items():
        if key == 'self':
            continue

        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)

        default = value.default if value.default is not inspect._empty else None
        defaults[key] = default

    return defaults

