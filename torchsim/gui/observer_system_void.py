import matplotlib.pyplot as plt
import numpy as np
from typing import List

from torchsim.gui.observables import Observable
from torchsim.gui.observer_system import ObserverPropertiesItem, ObserverSystemBase, MemoryBlockObservable, \
    ImageObservable, TextObservable, MatplotObservable, PropertiesObservable
from torchsim.gui.observers.cluster_observer import ClusterObserver
from torchsim.gui.observers.buffer_observer import BufferObserver
from torchsim.gui.observers.hierarchical_observer import HierarchicalObserver

plt.switch_backend("Qt5Agg")


class ObserverSystemVoid(ObserverSystemBase):
    def register_observer_hierarchical(self, name: str, observable: HierarchicalObserver):
        pass

    def register_observer_cluster(self, name: str, observable: ClusterObserver):
        pass

    def unregister_observer(self, name, close=False):
        pass

    def is_observer_registered(self, name: str) -> bool:
        pass

    def register_observer_matplot(self, name: str, observable: MatplotObservable):
        pass

    def register_observer_memory_block(self, name: str, observable: MemoryBlockObservable):
        pass

    def register_observer_image(self, name: str, observable: ImageObservable):
        pass

    def register_observer_text(self, name: str, observable: TextObservable):
        pass

    def register_observer_properties(self, name: str, observable: PropertiesObservable):
        pass

    def show_observer_image(self, name: str, data: np.ndarray):
        pass

    def show_observer_text(self, name: str, data: str):
        pass

    def show_observer_properties(self, name: str, properties: List[ObserverPropertiesItem]):
        pass

    def register_observer_buffer(self, name: str, observable: BufferObserver):
        pass

    def save_model_values(self):
        pass

    def load_model_values(self):
        pass

    def persist_observer_values(self, observer_name: str, observable: Observable):
        pass
