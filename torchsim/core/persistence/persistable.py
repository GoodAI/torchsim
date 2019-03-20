from abc import ABC, abstractmethod

from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver


class Persistable(ABC):
    @abstractmethod
    def save(self, parent_saver: Saver):
        pass

    @abstractmethod
    def load(self, parent_loader: Loader):
        pass
