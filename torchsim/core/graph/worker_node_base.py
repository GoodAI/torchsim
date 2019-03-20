import logging

from itertools import chain

from abc import abstractmethod, ABC

from torchsim.core.graph.invertible_node import InvertibleNode
from torchsim.core.graph.node_base import NodeInitializationException, NodeBase, TInputs, TOutputs, TInternals, EmptyOutputs
from torchsim.core.graph.unit import Unit
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver

logger = logging.getLogger(__name__)


class WorkerNodeWithInternalsBase(NodeBase[TInputs, TInternals, TOutputs], ABC):
    """Defines a Node with a Unit in the Graph.

    This class should be used as a base for all nodes which do some work (as opposed to node groups which are mostly
    containers for subgraphs). The Unit is agnostic to the graph and the connections - the node should handle those.

    If the unit needs dimensions of input tensors changed, do that in the step() method. When filling in memory_blocks
    or Outputs, reverse this viewing so that the node can be used in the graph consistently.


    If your node does not have internal memory_blocks, use WorkerNodeBase for convenience.

    Usage:
    CustomNode(WorkerNodeWithInternalsBase[CustomInputs, CustomInternals, CustomOutputs]):
        ...
    """

    _unit: Unit = None

    # The type hints are updated here.
    inputs: TInputs
    memory_blocks: TInternals
    outputs: TOutputs

    _last_dimensions = None

    def _all_memory_blocks(self):
        return chain(self.memory_blocks, self.outputs)

    @abstractmethod
    def _create_unit(self, creator: TensorCreator) -> Unit:
        pass

    def _fill_memory_blocks(self):
        """This prepares the slots after the unit is created."""
        self.memory_blocks.prepare_slots(self._unit)
        self.outputs.prepare_slots(self._unit)

    def _prepare_unit(self, tensor_creator):
        self._unit = self._create_unit(tensor_creator)
        self._fill_memory_blocks()
        self._check_slots()
        self._on_initialization_change()

    def _check_slots(self):
        """Checks that all memory block have tensors associated with them."""
        for block in self._all_memory_blocks():
            if block.tensor is None:
                message = f"Memory block '{block.name}' of node '{self.name}' does not have a tensor"
                raise NodeInitializationException(message)

    def allocate_memory_blocks(self, tensor_creator: TensorCreator):
        """Prepares the unit for the last time before the simulation is run.

        This will receive the AllocatingCreator.
        """
        self._prepare_unit(tensor_creator)

    def release_memory_blocks(self):
        """Release all allocated memory.

        Unit and memory blocks are released
        """
        self._unit = None
        self.memory_blocks.clear()
        self.outputs.clear()
        self._on_initialization_change()

    def detect_dims_change(self) -> bool:
        new_dimensions = [block.tensor.shape for block in self._all_memory_blocks()]
        changed = self._last_dimensions is None or self._last_dimensions != new_dimensions
        self._last_dimensions = new_dimensions

        return changed

    def _get_persistence_name(self):
        return f'{self.name}_{self.id}'

    def _save(self, saver: Saver):
        self._unit.save(saver)

    def _load(self, loader: Loader):
        self._unit.load(loader)

    def is_initialized(self) -> bool:
        return self._unit is not None

    def _on_initialization_change(self):
        """Unit state changed - it was created or destroyed"""
        pass


class WorkerNodeBase(WorkerNodeWithInternalsBase[TInputs, EmptyOutputs, TOutputs], ABC):
    """A base for worker nodes which do not have internal memory_blocks.

    See WorkerNodeWithInternalsBase for detailed documentation.
    """

    pass


class InvertibleWorkerNodeWithInternalsBase(WorkerNodeWithInternalsBase[TInputs, TInternals, TOutputs], InvertibleNode,
                                            ABC):
    """A base for worker nodes which have internal memory_blocks and support inverse projection.

    See WorkerNodeWithInternalsBase for detailed documentation of the worker node.
    """

    pass


class InvertibleWorkerNodeBase(InvertibleWorkerNodeWithInternalsBase[TInputs, EmptyOutputs, TOutputs], ABC):
    """A base for worker nodes which support inverse projection, but do not have internal memory_blocks.

    See WorkerNodeWithInternalsBase for detailed documentation of the worker node.
    """
    pass
