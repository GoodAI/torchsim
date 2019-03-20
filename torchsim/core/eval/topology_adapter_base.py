from abc import abstractmethod, ABC

from torchsim.core.graph import Topology


class TopologyAdapterBase(ABC):
    """A subject of the experiment (something like an adapter for the model for the particular experiment)."""

    @abstractmethod
    def get_topology(self) -> Topology:
        pass

    @abstractmethod
    def set_topology(self, topology: Topology):
        pass


class TestableTopologyAdapterBase(TopologyAdapterBase, ABC):
    """TopologyAdapterBase with switch_to_training() and switch_to_testing() methods."""

    @abstractmethod
    def is_in_training_phase(self, **kwargs) -> bool:
        pass

    @abstractmethod
    def switch_to_training(self):
        pass

    @abstractmethod
    def switch_to_testing(self):
        pass
