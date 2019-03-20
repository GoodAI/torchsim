from abc import abstractmethod

from torchsim.core.persistence.persistable import Persistable


class TrainTestSwitchable(Persistable):
    """A base class for topological topologies which allow for train/test mode switching."""
    @abstractmethod
    def switch_to_training(self):
        pass

    @abstractmethod
    def switch_to_testing(self):
        pass
