from abc import ABC, abstractmethod

from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorOutputs, SpaceEngineersConnectorInputs


class SeIoBase(ABC):
    """A common interface to the SE-(game/dataset0/dataset1).

    Provides ability to easily install the nodes into the topology and provide SE inputs and outputs.
    """
    outputs: SpaceEngineersConnectorOutputs
    inputs: SpaceEngineersConnectorInputs

    def install_nodes(self, target_group: NodeGroupBase):
        self._create_and_add_nodes()
        self._add_nodes(target_group)
        self._connect_nodes()

    @abstractmethod
    def _create_and_add_nodes(self):
        """Override this in order to add more nodes."""
        pass

    @abstractmethod
    def _add_nodes(self, target_group: NodeGroupBase):
        """Add nodes to the target_topology."""
        pass

    @abstractmethod
    def _connect_nodes(self):
        """Connect nodes between each other if necessary (architecture connected elsewhere)."""
        pass

    @abstractmethod
    def get_num_labels(self):
        """Return number of labels (if supported). This has to be accessible even before the sim. starts!"""
        pass

    @abstractmethod
    def get_image_numel(self):
        """Return number of elements in the image. Has to be accessible before first step!"""

    @abstractmethod
    def get_image_width(self):
        """ """

    @abstractmethod
    def get_image_height(self):
        """ """

    @abstractmethod
    def get_task_id(self) -> float:
        """Return the current task ID."""

    @abstractmethod
    def get_task_instance_id(self) -> float:
        """ """

    @abstractmethod
    def get_task_status(self) -> float:
        """ """

    @abstractmethod
    def get_task_instance_status(self) -> float:
        """ """

    @abstractmethod
    def get_reward(self) -> float:
        """ """

    @abstractmethod
    def get_testing_phase(self) -> float:
        """ """

    @abstractmethod
    def is_in_training_phase(self) -> bool:
        pass
