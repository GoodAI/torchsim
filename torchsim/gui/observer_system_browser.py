import torch

from abc import abstractmethod
import threading
from functools import partial
from typing import Dict, List, Callable, TypeVar, Generic

import numpy as np
import matplotlib.pyplot as plt

from torchsim.core.experiment_runner import UiExperimentRunner
from torchsim.gui.server.ui_api import UIApi, MemoryBlockParams
from torchsim.gui.server.ui_helper import PropertiesHolder, UIHelper
from torchsim.gui.observables import ObserverCallbacks
from torchsim.gui.observer_system import Observable, ObserverPropertiesItem, ObserverSystemBase, MemoryBlockObservable, \
    ImageObservable, TextObservable, MatplotObservable, PropertiesObservable, ObserverSystem
from torchsim.gui.observers.buffer_observer import BufferObserver, BufferObserverData
from torchsim.gui.observers.hierarchical_observer import HierarchicalObserver, HierarchicalObservableData
from torchsim.gui.observers.cluster_observer import ClusterObserver, ClusterObserverData
from torchsim.gui.observers.tensor_observable import TensorObservableData
import logging

from torchsim.gui.server.ui_server_connector import EventData
from torchsim.utils.dict_utils import to_nested_dict

plt.switch_backend("Qt5Agg")
logger = logging.getLogger(__name__)


class Observer:
    @abstractmethod
    def run(self, ui_helper: UIHelper):
        pass


class ImagePushObserver(Observer):

    def __init__(self, name: str, data: np.ndarray):
        self.data = data.copy()  # maybe not necessary
        self.name = name

    def run(self, ui_helper: UIHelper):
        ui_helper.ui_api.image(self.data, win=f'{self.name}')


T = TypeVar('T')


class PullObserver(Observer, Generic[T]):
    name: str
    observable: Observable

    def __init__(self, name: str, observable: Observable):
        self.observable = observable
        self.name = name

    def run(self, ui_helper: UIHelper):
        try:
            data = self.observable.get_data()
            properties = self.observable.get_properties()
            callbacks = self.observable.get_callbacks()
            self._run(ui_helper, data, properties, callbacks)
        except Exception:
            UiExperimentRunner._log_last_exception()

    @abstractmethod
    def _run(self, ui_helper: UIHelper, data: T, properties: List[ObserverPropertiesItem],
             callbacks: ObserverCallbacks):
        raise NotImplementedError


class MatplotPullObserver(PullObserver):

    def _run(self, ui_helper: UIHelper, plot: plt, properties: List[ObserverPropertiesItem],
             callbacks: ObserverCallbacks):
        ui_helper.ui_api.matplot(plot, win=f'{self.name}')


class RegisteringPullObserverBase(PullObserver[T], Generic[T]):
    """PullObserver registering properties and request callbacks."""
    observer_registered = False

    def __init__(self, name: str, observable: Observable, observer_system: ObserverSystem):
        super().__init__(name, observable)
        self._observer_system = observer_system
        self._properties_holder = PropertiesHolder([])

    def _run(self, ui_helper: UIHelper, data: TensorObservableData, properties: List[ObserverPropertiesItem],
             callbacks: ObserverCallbacks):
        self._properties_holder.properties = properties

        self._call_ui(ui_helper, data, properties)

        if not self.observer_registered:
            properties_callback = ui_helper.get_properties_callback(self.name, self._properties_holder, no_update=True)
            self.observer_registered = True
            ui_helper.clear_callbacks(self.name)

            def event_callback(event: EventData):
                logger.debug(f"Event: {event.event_type}")
                if event.event_type == 'property_updated':
                    properties_callback(event)
                elif event.event_type == 'window_closed':
                    self._observer_system.signals.window_closed.emit(self.name)

            ui_helper.register_event_callback(self.name, event_callback)

            for cb in callbacks.callbacks:
                if cb.command == 'event':
                    ui_helper.register_event_callback(self.name, cb.callback)
                elif cb.command == 'request':
                    ui_helper.register_request_callback(self.name, cb.callback)

    @abstractmethod
    def _call_ui(self, ui_helper: UIHelper, data: T, properties: List[ObserverPropertiesItem]):
        pass


class MemoryBlockPullObserver(RegisteringPullObserverBase[TensorObservableData]):

    def _call_ui(self, ui_helper: UIHelper, data: TensorObservableData, properties: List[ObserverPropertiesItem]):
        params = MemoryBlockParams(data.params.scale, data.params.projection, None)
        ui_helper.ui_api.memory_block(data.tensor, params,
                                      UIHelper.convert_properties(properties),
                                      win=f'{self.name}')


class BufferPullObserver(RegisteringPullObserverBase[BufferObserverData]):

    def _call_ui(self, ui_helper: UIHelper, data: BufferObserverData, properties: List[ObserverPropertiesItem]):
        params = MemoryBlockParams(data.tensor_data.params.scale, data.tensor_data.params.projection, data.current_ptr)
        ui_helper.ui_api.memory_block(data.tensor_data.tensor, params,
                                      UIHelper.convert_properties(properties),
                                      win=f'{self.name}')


class HierarchicalPullObserver(RegisteringPullObserverBase[HierarchicalObservableData]):

    def _call_ui(self, ui_helper: UIHelper, data: HierarchicalObservableData, properties: List[ObserverPropertiesItem]):
        ui_helper.ui_api.hierarchical_observer(data.groups_stacking, data.items_per_row, data.image_groups,
                                               [to_nested_dict(group_params) for group_params in data.params_groups],
                                               UIHelper.convert_properties(properties),
                                               win=f'{self.name}')


class ClusterPullObserver(RegisteringPullObserverBase[ClusterObserverData]):

    def _call_ui(self, ui_helper: UIHelper, data: ClusterObserverData, properties: List[ObserverPropertiesItem]):
        ui_helper.ui_api.cluster_observer(data,
                                          UIHelper.convert_properties(properties),
                                          win=self.name)


class ImagePullObserver(PullObserver):
    def _run(self, ui_helper: UIHelper, data: torch.Tensor, properties: List[ObserverPropertiesItem],
             callbacks: ObserverCallbacks):
        ui_helper.ui_api.image(data, win=f'{self.name}')


class TextPullObserver(PullObserver):

    def _run(self, ui_helper: UIHelper, data, properties: List[ObserverPropertiesItem], callbacks: ObserverCallbacks):
        ui_helper.ui_api.text(data, win=f'{self.name}')


class PropertiesPullObserver(PullObserver):

    def _run(self, ui_helper: UIHelper, data, properties: List[ObserverPropertiesItem], callbacks: ObserverCallbacks):
        ui_helper.properties(win=self.name, items=self.observable.get_properties())


class TextPushObserver(Observer):

    def __init__(self, name: str, data: str):
        self.data = data
        self.name = name

    def run(self, ui_helper: UIHelper):
        ui_helper.ui_api.text(self.data, win=f'{self.name}')


class ObserverSystemBrowser(ObserverSystemBase):
    _ui_api: UIApi

    update_period: float  # GUI update period [s]
    _observers_push: Dict[str, Observer] = dict()
    _observers_pull: Dict[str, PullObserver] = dict()
    _actions_to_run: List[Callable] = []

    def __init__(self, update_period: float = 0.1, storage_file: str = None):
        super().__init__(storage_file)
        self._ui_api = self._connect()
        self._ui_helper = UIHelper(self._ui_api, self.observer_persistence)
        self.update_period = update_period
        self._should_stop = False
        self._start()

    def stop(self):
        super().stop()
        self._should_stop = True

    def start(self):
        super().start()
        self._should_stop = False
        self._start()

    def unregister_observer(self, name, close=False):
        found_element = self._observers_pull.pop(name, None)

        def close_f(win: str):
            self._ui_api.close(win)

        if close and found_element is not None:
            self._actions_to_run.append(partial(close_f, name))

    def is_observer_registered(self, name: str) -> bool:
        return name in self._observers_pull

    def register_observer_matplot(self, name: str, observable: MatplotObservable):
        self._observers_pull[name] = MatplotPullObserver(name, observable)

    def register_observer_memory_block(self, name: str, observable: MemoryBlockObservable):
        self._observers_pull[name] = MemoryBlockPullObserver(name, observable, self)

    def register_observer_buffer(self, name: str, observable: BufferObserver):
        self._observers_pull[name] = BufferPullObserver(name, observable, self)

    def register_observer_image(self, name: str, observable: ImageObservable):
        self._observers_pull[name] = ImagePullObserver(name, observable)

    def register_observer_text(self, name: str, observable: TextObservable):
        self._observers_pull[name] = TextPullObserver(name, observable)

    def register_observer_properties(self, name: str, observable: PropertiesObservable):
        self._observers_pull[name] = PropertiesPullObserver(name, observable)

    def register_observer_hierarchical(self, name: str, observable: HierarchicalObserver):
        self._observers_pull[name] = HierarchicalPullObserver(name, observable, self)

    def register_observer_cluster(self, name: str, observable: ClusterObserver):
        self._observers_pull[name] = ClusterPullObserver(name, observable, self)

    def show_observer_image(self, name: str, data: np.ndarray):
        self._observers_push[name] = ImagePushObserver(name, data)

    def show_observer_text(self, name: str, data: str):
        self._observers_push[name] = TextPushObserver(name, data)

    def _start(self):
        self._periodic_loop()

    def _periodic_loop(self):
        if self._should_stop:
            return

        observers_push = list(self._observers_push.values())
        self._observers_push.clear()

        observers_pull = list(self._observers_pull.values())
        actions = list(self._actions_to_run)
        self._actions_to_run.clear()

        for observer in observers_pull:
            observer.run(self._ui_helper)

        for observer in observers_push:
            observer.run(self._ui_helper)

        for action in actions:
            action()
        threading.Timer(self.update_period, self._periodic_loop).start()

    @staticmethod
    def _connect() -> UIApi:
        # Connect to UI server
        connector = UIApi(server='ws://localhost', port=5000)
        connector.remove_all_windows()
        return connector

    def save_model_values(self):
        for name, observer in self._observers_pull.items():
            self.observer_persistence.store_values(name, observer.observable)

    def load_model_values(self):
        for name, observer in self._observers_pull.items():
            self.observer_persistence.read_stored_values(name, observer.observable, True)

    def persist_observer_values(self, observer_name: str, observable: Observable):
        if self.observer_persistence is not None:
            self.observer_persistence.store_values(observer_name, observable)

