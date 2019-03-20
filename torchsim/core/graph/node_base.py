import logging
from collections import OrderedDict

from itertools import chain
from typing import List, Dict, Generic, TypeVar, Optional

from abc import ABC, abstractmethod

from torchsim.core.graph.slot_container_base import InputsBase, SlotContainerBase, \
    OutputsBase
from torchsim.core.graph.slots import NoneSlot
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.model import PropertiesProvider, ObservableProvider
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.persistable import Persistable
from torchsim.core.persistence.saver import Saver
from torchsim.gui.observables import Observable, ObserverPropertiesItem, Initializable, ObserverPropertiesBuilder, \
    ObserverPropertiesItemSourceType
from torchsim.utils.cache_utils import SimpleResettableCache

logger = logging.getLogger(__name__)


class NodeInitializationException(Exception):
    pass


class NodeValidationException(Exception):
    pass


TInputs = TypeVar('TInputs', bound=InputsBase)
TInternals = TypeVar('TInternals', bound=OutputsBase)
TOutputs = TypeVar('TOutputs', bound=OutputsBase)


class NodeBase(PropertiesProvider, ObservableProvider, Persistable, ABC, Generic[TInputs, TInternals, TOutputs],
               Initializable):
    """Defines a basic Node in the Graph.

    This should generally not be subclassed - look at worker_node_base or node_group modules instead.

    Usage:
    class ReallyCustomNode(NodeBase[SuchCustomInputs, MuchCustomInternals, VeryCustomOutputs]):
        ...
    """
    topological_order: int

    _name: str
    _id: int

    # All inputs.
    inputs: TInputs
    # inputs: InputsBase[TInputSlot]
    # Memory blocks which are normally not used as outputs.
    memory_blocks: TInternals
    # memory_blocks: OutputsBase[TOutputSlot]
    # Memory blocks which are normally used as outputs.
    outputs: TOutputs
    # outputs: OutputsBase[TOutputSlot]

    _skip: bool
    _prop_builder: ObserverPropertiesBuilder
    _single_step_scoped_cache: SimpleResettableCache

    @property
    def name(self):
        if self._name is None:
            self._name = "Node"

        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def name_with_id(self):
        return f"#{self._id} {self.name}"

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def skip_execution(self) -> bool:
        return self._skip

    @skip_execution.setter
    def skip_execution(self, value: bool):
        self._skip = value

    def __init__(self, name: str = None,
                 inputs: TInputs = None,
                 memory_blocks: TOutputs = None,
                 outputs: TOutputs = None):
        """Initializes the node.

        Inputs, memory_blocks (== internals) and outputs should be initialized here and accessible from now on
        for connecting.
        """
        # TODO (Feat): Auto-name nodes as in BrainSimulator, or remove the default value of parameter 'name'.
        self._name = name
        self.topological_order = -1
        self._id = 0
        self._skip = False

        self.inputs = inputs if inputs is not None else EmptyInputs(self)
        self.memory_blocks = memory_blocks if memory_blocks is not None else EmptyOutputs(self)
        self.outputs = outputs if outputs is not None else EmptyOutputs(self)
        self._prop_builder = ObserverPropertiesBuilder(self, source_type=ObserverPropertiesItemSourceType.MODEL)
        self._single_step_scoped_cache = SimpleResettableCache()

    def get_observables(self) -> Dict[str, Observable]:
        """Collect things that can be observed.

        Do not override this method, override _get_observables() instead!
        """
        observables = chain(self._get_memory_block_observables().items(), self._get_observables().items())

        # Add prefix to observable names
        prefixed_observables: OrderedDict[str, Observable] = OrderedDict()
        for name, observer in observables:
            prefixed_observables[f"{self.name_with_id}.{name}"] = observer
        return prefixed_observables

    def _get_memory_block_observables(self) -> Dict[str, Observable]:
        def create_observers(result: Dict[str, Observable], prefix: str, container: SlotContainerBase):
            for item in container:
                result[f'{prefix}.{item.name}'] = item.get_observable()

        observables: OrderedDict[str, Observable] = OrderedDict()
        create_observers(observables, 'Input', self.inputs)
        create_observers(observables, 'Internal', self.memory_blocks)
        create_observers(observables, 'Output', self.outputs)
        return observables

    def _get_observables(self) -> Dict[str, Observable]:
        """Get observables of the node.

        Override this method in subclasses to add custom Observables.

        Returns:
            Dict of name -> Observable.
        """
        return {}

    @abstractmethod
    def detect_dims_change(self) -> bool:
        """Checks whether the dimensions of slots have changed since last time this was called.

        Check the change of the inputs here (if something changed, recompute output dimensions).

        Returns:
            True if the sizes changed, False otherwise.
        """
        pass

    @abstractmethod
    def allocate_memory_blocks(self, tensor_creator: TensorCreator):
        """Prepares the unit before the simulation is run.

        This gets called multiple times during the dimension "shake-down" and then once before the simulation runs.
        """
        pass

    @abstractmethod
    def release_memory_blocks(self):
        """Revert the unit to the uninitialized state.

        Release the unit, and all memory blocks.
        """
        pass

    def validate(self):
        """Called after allocate_memory_blocks, before the first step runs.

        If a node cannot run in the current configuration of state/connected inputs/tensor dimensions, it should raise
        NodeValidationException (or a subclass).
        """
        pass

    def step(self):
        """Perform one node step unless self.skip_execution is True."""
        if not self.skip_execution:
            self._single_step_scoped_cache.reset()
            self._step()

    @abstractmethod
    def _step(self):
        """Perform the execution of the step.

        This should retrieve input tensors and pass them into unit.step().
        """

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [self._prop_builder.auto('Skip execution', type(self).skip_execution)]

    def _get_persistence_name(self):
        return f'{self.name}_{self.id}'

    def save(self, parent_saver: Saver, persistence_name: Optional[str] = None):
        """Save the node in the context of its parent saver."""

        folder_name = persistence_name or self._get_persistence_name()
        saver = parent_saver.create_child(folder_name)

        self._save(saver)

    def load(self, parent_loader: Loader):
        """Load the node and its tensors from location relative to the parent loader."""
        folder_name = self._get_persistence_name()
        loader = parent_loader.load_child(folder_name)

        self._load(loader)

    def _save(self, saver: Saver):
        pass

    def _load(self, loader: Loader):
        pass


class EmptyInputs(InputsBase[NoneSlot, NodeBase]):
    """This is the default instance of Inputs, used in nodes which don't have inputs."""

    def _create_slot_instance(self, name) -> NoneSlot:
        raise RuntimeError("No slots can be added to EmptyInputs")


class EmptyOutputs(OutputsBase[NoneSlot, NodeBase]):
    """This is the default instance of Outputs, used in nodes which don't have outputs."""

    def _create_slot_instance(self, name) -> NoneSlot:
        raise RuntimeError("No slots can be added to EmptyOutputs")