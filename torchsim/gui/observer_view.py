import logging
from functools import partial
from typing import Dict, List

from torchsim.gui.observables import ObserverPropertiesBuilder
from torchsim.gui.observer_system import ObserverSystem, Observable, ObserverPropertiesItem, PropertiesObservable

logger = logging.getLogger(__name__)


class ObserverView(PropertiesObservable):
    """A node that encompasses all the model's observables and passes them on to the observer system."""
    _strip_observer_name_prefix: str

    _observables: Dict[str, Observable]
    _first_show: bool = True

    def __init__(self, name: str, observer_system: ObserverSystem, strip_observer_name_prefix: str = ''):
        self._strip_observer_name_prefix = strip_observer_name_prefix
        self.name = name
        self._observer_system = observer_system
        self._observables = {}
        observer_system.signals.window_closed.connect(self.on_window_closed)
        self._prop_builder = ObserverPropertiesBuilder(self)

    def _persist(self):
        self._observer_system.persist_observer_values(self.name, self)

    def on_window_closed(self, observer_name: str):
        if observer_name in self._observables:
            self._observer_system.unregister_observer(observer_name, False)
            self._persist()

    def close(self):
        self._unregister_observers()
        self._observer_system.unregister_observer(self.name, True)

    def set_observables(self, observables: Dict[str, Observable]):
        self._unregister_observers()
        self._observables = observables
        # default is no observers visible
        # self._register_observers()
        if self._first_show:
            self._observer_system.register_observer(self.name, self)
            self._first_show = False

    def _register_observers(self):
        for name, observable in self._observables.items():
            self._observer_system.register_observer(name, observable)

    def _unregister_observers(self):
        for name in self._observables.keys():
            self._observer_system.unregister_observer(name, True)

    def get_properties(self) -> List[ObserverPropertiesItem]:
        def enable_observers_handler(prop_name: str, value: bool):
            if value:
                logger.debug(f"Register observer {name}")
                self._observer_system.register_observer(prop_name, self._observables[prop_name])
            else:
                logger.debug(f"Unregister observer {name}")
                self._observer_system.unregister_observer(prop_name, True)

        def remove_prefix(text: str, prefix: str):
            if text.startswith(prefix):
                return text[len(prefix):]
            else:
                return text

        observers = []
        last_header = ''
        for name, observable in self._observables.items():
            observer_name = remove_prefix(name, self._strip_observer_name_prefix)
            header = observer_name.split('.')[0]
            observer_name = remove_prefix(observer_name, f'{header}.')
            # add collapsible_header
            if last_header != header:
                last_header = header
                observers.append(self._prop_builder.collapsible_header(header, False))

            observers.append(self._prop_builder.checkbox(
                observer_name,
                self._observer_system.is_observer_registered(name),
                partial(enable_observers_handler, name)))

        def set_all():
            self._register_observers()
            self._persist()

        def set_none():
            self._unregister_observers()
            self._persist()

        return [
                    self._prop_builder.button('All', set_all),
                    self._prop_builder.button('None', set_none),
               ] + observers
