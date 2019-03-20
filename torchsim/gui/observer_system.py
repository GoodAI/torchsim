import os
import logging
import threading

from abc import abstractmethod, ABC
from collections import OrderedDict

from typing import Any, List, Dict, Optional

import numpy as np
from ruamel.yaml import YAML, YAMLError

from torchsim.core.exceptions import IllegalStateException, FailedValidationException
from torchsim.core.utils.singals import signal
from torchsim.gui.observables import Observable, ObserverPropertiesItem, MemoryBlockObservable, \
    ImageObservable, TextObservable, MatplotObservable, PropertiesObservable, ObserverPropertiesItemState, \
    ObserverPropertiesItemType, ObserverPropertiesItemSourceType
from torchsim.gui.observers.buffer_observer import BufferObserver
from torchsim.gui.observers.hierarchical_observer import HierarchicalObserver
from torchsim.utils.os_utils import last_exception_as_html

logger = logging.getLogger(__name__)


class ClusterObservable(Observable):
    @abstractmethod
    def get_data(self) -> None:
        pass

    @abstractmethod
    def get_properties(self) -> List['ObserverPropertiesItem']:
        pass


class ObserverSystemSignals:
    window_closed = signal(str)


class ObserverSystem(ABC):
    # static field - lock to ensure just one observer system at the time can be instantiated
    initialized: bool = False
    signals: ObserverSystemSignals

    def __init__(self):
        if ObserverSystem.initialized:
            raise IllegalStateException('Observer system already instantiated. Cannot be run multiple times')
        ObserverSystem.initialized = True
        self.signals = ObserverSystemSignals()

    @abstractmethod
    def register_observer(self, name: str, observable: Observable):
        pass

    @abstractmethod
    def unregister_observer(self, name, close=False):
        pass

    @abstractmethod
    def is_observer_registered(self, name: str) -> bool:
        pass

    @abstractmethod
    def save_model_values(self):
        pass

    @abstractmethod
    def load_model_values(self):
        pass

    @abstractmethod
    def persist_observer_values(self, observer_name: str, observable: Observable):
        pass


class ObserverSystemBase(ObserverSystem):
    observer_persistence: 'Optional[ObserverPersistence]' = None

    def __init__(self, storage_file: str = None):
        super().__init__()
        if storage_file is not None:
            self.observer_persistence = SimpleFileObserverPersistence(storage_file)

    def register_observer(self, name: str, observable: Observable):
        def not_found(_, _observable):
            raise ValueError(f'Observer type "{type(_observable)}" is not supported')

        switch = OrderedDict([
            (BufferObserver, self.register_observer_buffer),
            (MemoryBlockObservable, self.register_observer_memory_block),
            (ImageObservable, self.register_observer_image),
            (TextObservable, self.register_observer_text),
            (MatplotObservable, self.register_observer_matplot),
            (PropertiesObservable, self.register_observer_properties),
            (HierarchicalObserver, self.register_observer_hierarchical),
            (ClusterObservable, self.register_observer_cluster),
            (Observable, not_found)
        ])

        if self.observer_persistence is not None:
            self.observer_persistence.set_default_values(name, observable)
            self.observer_persistence.read_stored_values(name, observable)

        for t, func in switch.items():
            if isinstance(observable, t):
                func(name, observable)
                break

    def start(self):
        if self.observer_persistence is not None:
            self.observer_persistence.start()

    def stop(self):
        if self.observer_persistence is not None:
            self.observer_persistence.stop()

    @abstractmethod
    def save_model_values(self):
        pass

    @abstractmethod
    def load_model_values(self):
        pass

    @abstractmethod
    def unregister_observer(self, name, close=False):
        raise NotImplementedError

    @abstractmethod
    def is_observer_registered(self, name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def register_observer_properties(self, name: str, observable: PropertiesObservable):
        raise NotImplementedError

    @abstractmethod
    def register_observer_matplot(self, name: str, observable: MatplotObservable):
        raise NotImplementedError

    @abstractmethod
    def register_observer_memory_block(self, name: str, observable: MemoryBlockObservable):
        raise NotImplementedError

    @abstractmethod
    def register_observer_buffer(self, name: str, observable: BufferObserver):
        raise NotImplementedError

    @abstractmethod
    def register_observer_image(self, name: str, observable: ImageObservable):
        raise NotImplementedError

    @abstractmethod
    def register_observer_text(self, name: str, observable: TextObservable):
        raise NotImplementedError

    @abstractmethod
    def register_observer_cluster(self, name: str, observable: ClusterObservable):
        raise NotImplementedError

    @abstractmethod
    def register_observer_hierarchical(self, name: str, observable: HierarchicalObserver):
        raise NotImplementedError

    @abstractmethod
    def show_observer_image(self, name: str, data: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def show_observer_text(self, name: str, data: str):
        raise NotImplementedError


class ObserverPersistence:
    @abstractmethod
    def set_default_values(self, observer_name: str, observable: Observable):
        pass

    @abstractmethod
    def read_stored_values(self, observer_name: str, observable: Observable, load_model_values: bool = False):
        pass

    @abstractmethod
    def store_value(self, observer_name: str, item_name: str, item_value: Any):
        pass

    def store_values(self, observer_name: str, observable: Observable):
        for prop in observable.get_properties():
            self.store_value(observer_name, prop.name, prop.value)

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class SimpleFileObserverPersistence(ObserverPersistence):
    _storage_file: str
    _values: Dict[str, Dict[str, Any]]
    _should_write_file: bool = False  # set to true to write file by background thread
    WRITE_PERIOD: float = 1.0  # minimal period to write to file
    _should_stop: bool = False

    def __init__(self, storage_file: str):
        self._yaml = YAML()
        self._storage_file = storage_file

        # Init values
        if os.path.isfile(self._storage_file):
            self._values = self._read_from_file()
        else:
            self._values = OrderedDict()

        # self.start_writer_thread()
        self.start()

    def read_stored_values(self, observer_name: str, observable: Observable, load_model_items: bool = False):
        indexed_properties: Dict[str, ObserverPropertiesItem] = {item.name: item for item in
                                                                 observable.get_properties()}
        if observer_name in self._values:
            for key, value in self._values[observer_name].items():
                if key in indexed_properties:
                    # values are sent as string
                    # ignore buttons
                    value_to_send = value
                    prop = indexed_properties[key]
                    if prop.type == ObserverPropertiesItemType.BUTTON \
                            or prop.state != ObserverPropertiesItemState.ENABLED \
                            or (prop.source_type == ObserverPropertiesItemSourceType.MODEL and not load_model_items):
                        continue
                    # elif prop.type == "checkbox":
                    #     value_to_send =
                    try:
                        prop.callback(value_to_send)
                    except (ValueError, FailedValidationException):
                        logger.error(f'Error in reading key <code>{observer_name}.{key}</code>, value "{value}":<br/>'
                                     f'{last_exception_as_html()}')

    def set_default_values(self, observer_name: str, observable: Observable):
        if observer_name not in self._values:
            self._values[observer_name] = OrderedDict()
        for item in observable.get_properties():
            if item.name not in self._values[observer_name]:
                self._values[observer_name][item.name] = item.value

    def store_value(self, observer_name: str, item_name: str, item_value: Any):
        if observer_name not in self._values:
            self._values[observer_name] = OrderedDict()
        self._values[observer_name][item_name] = item_value
        self._should_write_file = True

    def _write_to_file(self):
        logging.debug(f'Writing to file')
        # Dump YAML to temporary file and move it to the storage_file as atomic operation (to prevent file corruption)
        temp_file = f'{self._storage_file}.temp'
        with open(temp_file, 'w') as file:
            self._yaml.dump(self._values, file)
        os.replace(temp_file, self._storage_file)

    def _read_from_file(self) -> Dict[str, Dict[str, Any]]:
        with open(self._storage_file, 'r') as file:
            try:
                config = self._yaml.load(file)
            except AssertionError:
                logger.error(f"Error parsing persistence YAML file {self._storage_file}: {last_exception_as_html()}")
                config = None
            except YAMLError:
                logger.error(f"Error parsing persistence YAML file {self._storage_file}: {last_exception_as_html()}")
                config = None

            return config if config is not None else OrderedDict()

    def start(self):
        self._should_stop = False
        self._start()

    def stop(self):
        self._should_stop = True

    def _start(self):
        self._periodic_loop()

    def _periodic_loop(self):
        if self._should_stop:
            return

        if self._should_write_file:
            self._write_to_file()
            self._should_write_file = False

        threading.Timer(self.WRITE_PERIOD, self._periodic_loop).start()
