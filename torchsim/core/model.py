from abc import abstractmethod
from typing import Dict, List

from torchsim.gui.observables import Observable, ObserverPropertiesItem


class PropertiesProvider:
    """A node which has properties that can be adjusted from the UI."""
    @abstractmethod
    def get_properties(self) -> List[ObserverPropertiesItem]:
        raise NotImplementedError


class ObservableProvider:
    """A node which has observables (which can be observed from the UI)."""
    @abstractmethod
    def get_observables(self) -> Dict[str, Observable]:
        raise NotImplementedError


class Model(PropertiesProvider):
    """A model which can be simulated."""
    @abstractmethod
    def step(self):
        raise NotImplementedError

