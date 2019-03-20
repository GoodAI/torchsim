import numpy as np
from abc import abstractmethod
from enum import Enum, EnumMeta
from functools import partial
from typing import Any, List, Callable, NamedTuple, Dict, Union, TypeVar, Type, get_type_hints, Optional, Tuple, \
    Iterable
import matplotlib.pyplot as plt

from torchsim.core.exceptions import IllegalArgumentException, FailedValidationException
from torchsim.gui.server.ui_server_connector import RequestData, EventData
from torchsim.gui.ui_utils import parse_bool

T = TypeVar('T')


class Initializable:
    @abstractmethod
    def is_initialized(self) -> bool:
        pass


class ObserverPropertiesItemState(Enum):
    ENABLED = 1
    DISABLED = 2
    READ_ONLY = 3


class EditStrategyResult(NamedTuple):
    state: ObserverPropertiesItemState
    reason: Optional[str] = None


EditStrategy = Callable[[Initializable], Optional[EditStrategyResult]]


def check_type(expected_type: type, instance):
    if not isinstance(instance, expected_type):
        raise IllegalArgumentException(f'Expected instance of {expected_type} but {type(instance)} received.')


def enable_on_runtime(instance: Initializable) -> Optional[EditStrategyResult]:
    check_type(Initializable, instance)
    if not instance.is_initialized():
        return EditStrategyResult(ObserverPropertiesItemState.DISABLED,
                                  "Item can be modified only when the simulation is running")


def disable_on_runtime(instance: Initializable) -> Optional[EditStrategyResult]:
    check_type(Initializable, instance)
    if instance.is_initialized():
        return EditStrategyResult(ObserverPropertiesItemState.DISABLED,
                                  "Item can be modified only when the simulation is stopped")


class ObserverPropertiesItemSourceType(Enum):
    MODEL = 1,
    OBSERVER = 2,
    CONTROL = 3


class ObserverPropertiesItemType(Enum):
    TEXT = 1
    NUMBER = 2
    BUTTON = 3
    CHECKBOX = 4
    SELECT = 5
    COLLAPSIBLE_HEADER = 6


class ObserverPropertiesItemSelectValueItem(NamedTuple):
    name: str


class ObserverPropertiesItem:
    select_values: List[ObserverPropertiesItemSelectValueItem]
    STATE_MAPPING = {
        ObserverPropertiesItemState.ENABLED: 'enabled',
        ObserverPropertiesItemState.DISABLED: 'disabled',
        ObserverPropertiesItemState.READ_ONLY: 'readonly',
    }
    TYPE_MAPPING = {
        ObserverPropertiesItemType.TEXT: 'text',
        ObserverPropertiesItemType.NUMBER: 'number',
        ObserverPropertiesItemType.BUTTON: 'button',
        ObserverPropertiesItemType.CHECKBOX: 'checkbox',
        ObserverPropertiesItemType.SELECT: 'select',
        ObserverPropertiesItemType.COLLAPSIBLE_HEADER: 'collapsible_header',
    }
    SOURCE_TYPE_MAPPING = {
        ObserverPropertiesItemSourceType.MODEL: 'model',
        ObserverPropertiesItemSourceType.OBSERVER: 'observer',
        ObserverPropertiesItemSourceType.CONTROL: 'control'
    }

    type: ObserverPropertiesItemType
    name: str
    value: Any
    callback: Callable[[str], str]
    state: ObserverPropertiesItemState
    _reverse_type_dict: Dict[str, ObserverPropertiesItemType] = {v: k for k, v in TYPE_MAPPING.items()}
    optional: bool

    def __init__(self, name: str, type_name: str, value: Any, callback: Callable[[str], Optional[str]],
                 state: ObserverPropertiesItemState = ObserverPropertiesItemState.ENABLED, optional: bool = False,
                 select_values: List[ObserverPropertiesItemSelectValueItem] = (),
                 source_type: ObserverPropertiesItemSourceType = ObserverPropertiesItemSourceType.OBSERVER,
                 hint: str = ''
                 ):
        if type_name not in self.TYPE_MAPPING.values():
            raise IllegalArgumentException(f"Unrecognized type '{self.type}'")

        self.type = ObserverPropertiesItem._reverse_type_dict[type_name]
        self.name = name
        self.value = value
        self.callback = callback
        self.state = state
        self.select_values = select_values
        self.optional = optional
        self.source_type = source_type
        self.hint = hint
        self._check_arguments()

    def _check_arguments(self):
        if self.type == self.TYPE_MAPPING[ObserverPropertiesItemType.SELECT] and len(self.select_values) == 0:
            raise IllegalArgumentException(f"Select type must have select_values defined")

    @staticmethod
    def create(name: str, property_type: ObserverPropertiesItemType, value: str, callback: Callable[[str], str],
               state: ObserverPropertiesItemState = ObserverPropertiesItemState.ENABLED, optional: bool = False,
               select_values: List[ObserverPropertiesItemSelectValueItem] = (),
               source_type: ObserverPropertiesItemSourceType = ObserverPropertiesItemSourceType.OBSERVER,
               hint: str = ''
               ) -> 'ObserverPropertiesItem':
        return ObserverPropertiesItem(name, ObserverPropertiesItem.TYPE_MAPPING[property_type], value, callback, state,
                                      optional, select_values, source_type, hint)

    @staticmethod
    def clone_with_prefix(item: 'ObserverPropertiesItem', prefix: str) -> 'ObserverPropertiesItem':
        return ObserverPropertiesItem.create(f'{prefix}{item.name}', item.type, item.value, item.callback, item.state,
                                             item.optional, item.select_values, item.source_type, item.hint)


class ObserverCallbackItem(NamedTuple):
    command: str
    callback: Callable[[any], any]


class ObserverCallbacks:
    _callbacks: List[ObserverCallbackItem]

    def __init__(self) -> None:
        self._callbacks = []

    @property
    def callbacks(self):
        return self._callbacks

    def _add(self, command: str, callback: Callable[[any], any]):
        self._callbacks.append(ObserverCallbackItem(command, callback))

    def add_request(self, callback: Callable[[RequestData], any]) -> 'ObserverCallbacks':
        self._add('request', callback)
        return self

    def add_event(self, callback: Callable[[EventData], any]) -> 'ObserverCallbacks':
        self._add('event', callback)
        return self


class Observable:
    """Interface tagging objects observable within ObserverSystem.

    Note: Do not implement directly, use some of its subclasses
    """

    @abstractmethod
    def get_data(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_properties(self) -> List['ObserverPropertiesItem']:
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def get_callbacks(self) -> ObserverCallbacks:
        return ObserverCallbacks()


class MatplotObservable(Observable):
    @abstractmethod
    def get_data(self) -> plt:
        pass

    def get_properties(self) -> List['ObserverPropertiesItem']:
        return []


class ImageObservable(Observable):
    @abstractmethod
    def get_data(self) -> np.ndarray:
        pass

    def get_properties(self) -> List['ObserverPropertiesItem']:
        return []


class MemoryBlockObservable(Observable):
    @abstractmethod
    def get_data(self) -> np.ndarray:
        pass

    def get_properties(self) -> List['ObserverPropertiesItem']:
        return []


class TextObservable(Observable):
    @abstractmethod
    def get_data(self) -> str:
        pass

    def get_properties(self) -> List['ObserverPropertiesItem']:
        return []


class PropertiesObservable(Observable):
    def get_data(self) -> None:
        return None

    @abstractmethod
    def get_properties(self) -> List['ObserverPropertiesItem']:
        pass


class LambdaPropertiesObservable(PropertiesObservable):
    properties_provider: Callable[[], List[ObserverPropertiesItem]]

    def __init__(self, properties_provider: Callable[[], List['ObserverPropertiesItem']]):
        self.properties_provider = properties_provider

    def get_properties(self) -> List['ObserverPropertiesItem']:
        return self.properties_provider()


E = TypeVar('E', Enum, Enum)


class ObserverPropertiesBuilder:
    _source_type: ObserverPropertiesItemSourceType  # Model properties are not loaded from persistence automatically
    _instance: Optional[object]
    _collapsible_header_storage: Dict[str, bool]

    def __init__(self, instance: object = None,
                 source_type: ObserverPropertiesItemSourceType = ObserverPropertiesItemSourceType.OBSERVER):
        self._source_type = source_type
        self._collapsible_header_storage = {}
        self._instance = instance

    def collapsible_header(self, name: str, default_is_expanded: bool,
                           state: ObserverPropertiesItemState = ObserverPropertiesItemState.ENABLED) -> ObserverPropertiesItem:
        if name not in self._collapsible_header_storage:
            self._collapsible_header_storage[name] = default_is_expanded

        def update_value(v: Union[bool, str]):
            self._collapsible_header_storage[name] = parse_bool(v)
            return v

        return ObserverPropertiesItem(
            name,
            ObserverPropertiesItem.TYPE_MAPPING[ObserverPropertiesItemType.COLLAPSIBLE_HEADER],
            self._collapsible_header_storage[name],
            update_value,
            state
        )

    def checkbox(self, name: str, value: bool, update_cb: Callable[[bool], Union[bool, None]],
                 state: ObserverPropertiesItemState = ObserverPropertiesItemState.ENABLED):
        def update_value(v: Union[bool, str]):
            result = update_cb(parse_bool(v))
            return v if result is None else result

        return ObserverPropertiesItem(
            name,
            ObserverPropertiesItem.TYPE_MAPPING[ObserverPropertiesItemType.CHECKBOX],
            value,
            update_value,
            state
        )

    def number_int(self, name: str, value: int, update_cb: Callable[[int], Union[int, None]],
                   state: ObserverPropertiesItemState = ObserverPropertiesItemState.ENABLED):
        def update_value(v: Union[int, str]):
            if v is None:
                return 0
            else:
                result = update_cb(int(v))
                return v if result is None else result

        return ObserverPropertiesItem.create(name, ObserverPropertiesItemType.NUMBER, str(value), update_value, state)

    def select(self, name: str, value: E, update_cb: Callable[[E], Union[E, None]], select_values: Type[Enum],
               state: ObserverPropertiesItemState = ObserverPropertiesItemState.ENABLED):
        # Type hints don't like list(select_values).
        select_items = [value for value in select_values]

        def format_item(v: E):
            return select_items.index(v)

        def parse_item(v: Union[int, str]):
            return select_items[int(v)]

        def update_value(v: Union[int, str]):
            item = parse_item(v)
            update_cb(item)
            return str(format_item(item))

        return ObserverPropertiesItem(
            name,
            ObserverPropertiesItem.TYPE_MAPPING[ObserverPropertiesItemType.SELECT],
            format_item(value),
            update_value,
            state,
            select_values=[ObserverPropertiesItemSelectValueItem(e.name) for e in select_items],
        )

    @staticmethod
    def _create_update_value(setter: Callable[[T], None], parser: Optional[Callable[[str], T]]) -> Callable[[str], str]:
        def update_value(value: str):
            setter(parser(value) if parser else value)
            return value

        return update_value

    @staticmethod
    def _format_list(l: List[T], item_formatter: Callable[[T], str] = str) -> str:
        return ','.join(map(item_formatter, l))

    @staticmethod
    def _parse_list(text: str, item_parser: Callable[[str], T]) -> List[T]:
        return [] if len(text) == 0 else [item_parser(t) for t in text.split(',')]

    @staticmethod
    def _parse_enum(text: str, item_parser: Callable[[str], T]) -> List[T]:
        return [] if len(text) == 0 else [item_parser(t) for t in text.split(',')]

    def _parse_list_int(self, v: str) -> List[int]:
        try:
            return self._parse_list(v, lambda i: int(i))
        except ValueError as e:
            raise FailedValidationException(f"Expected List[int], syntax error: {e}")

    def _parse_list_float(self, v: str) -> List[float]:
        try:
            return self._parse_list(v, lambda i: float(i))
        except ValueError as e:
            raise FailedValidationException(f"Expected List[float], syntax error: {e}")

    def auto(self, name: str, prop: property,
             state: Optional[ObserverPropertiesItemState] = None,
             enabled: Optional[bool] = None,
             edit_strategy: Optional[EditStrategy] = None,
             hint: str = ''):

        def prop_tuple_int(count: int):
            def parse_tuple_int(v: str) -> Iterable[int]:
                try:
                    parsed_list = self._parse_list(v, lambda i: int(i))
                    if len(parsed_list) != count:
                        raise ValueError(f'Expected exactly {count} items, but {len(parsed_list)} received')
                    return tuple(parsed_list)
                except ValueError as e:
                    raise FailedValidationException(f"Expected Tuple[{','.join(['int'] * count)}], syntax error: {e}")

            return self.prop(name, prop, parse_tuple_int, self._format_list, ObserverPropertiesItemType.TEXT, state,
                             hint=hint)

        def prop_tuple_float(count: int):
            def parse_tuple_float(v: str) -> Iterable[float]:
                try:
                    parsed_list = self._parse_list(v, lambda i: float(i))
                    if len(parsed_list) != count:
                        raise ValueError(f'Expected exactly {count} items, but {len(parsed_list)} received')
                    return tuple(parsed_list)
                except ValueError as e:
                    raise FailedValidationException(f"Expected Tuple[{','.join(['float'] * count)}], syntax error: {e}")

            return self.prop(name, prop, parse_tuple_float, self._format_list, ObserverPropertiesItemType.TEXT, state,
                             hint=hint)

        tuple_mapping = {
            Tuple[int]: partial(prop_tuple_int, 1),
            Tuple[int, int]: partial(prop_tuple_int, 2),
            Tuple[int, int, int]: partial(prop_tuple_int, 3),
            Tuple[int, int, int, int]: partial(prop_tuple_int, 4),
            Tuple[int, int, int, int, int]: partial(prop_tuple_int, 5),
            Tuple[int, int, int, int, int, int]: partial(prop_tuple_int, 6),
            Tuple[float]: partial(prop_tuple_float, 1),
            Tuple[float, float]: partial(prop_tuple_float, 2),
            Tuple[float, float, float]: partial(prop_tuple_float, 3),
            Tuple[float, float, float, float]: partial(prop_tuple_float, 4),
            Tuple[float, float, float, float, float]: partial(prop_tuple_float, 5),
            Tuple[float, float, float, float, float, float]: partial(prop_tuple_float, 6),
        }

        def is_tuple_type(prop_type):
            for t in tuple_mapping.keys():
                if prop_type is t:
                    return True
            return False

        def make_tuple_prop(prop_type):
            for (t, f) in tuple_mapping.items():
                if prop_type is t:
                    return f()
            raise IllegalArgumentException(f'Unrecognized type {prop_type}')

        self._check_instance_is_set()
        state, state_reason = self._resolve_state_strategy(self._resolve_state(state, enabled), edit_strategy)

        hints = get_type_hints(prop.fget)
        if 'return' not in hints:
            raise IllegalArgumentException(f'Property getter must be annotated by a return value type hint')
        prop_type = hints['return']

        if prop_type is int:
            return self.prop(name, prop, int, None, ObserverPropertiesItemType.NUMBER, state, hint=hint)
        elif prop_type is str:
            return self.prop(name, prop, None, None, ObserverPropertiesItemType.TEXT, state, hint=hint)
        elif prop_type is Optional[int]:
            return self.prop(name, prop, lambda v: None if v is None else int(v), None,
                             ObserverPropertiesItemType.NUMBER, state, optional=True, hint=hint)
        elif prop_type is float:
            return self.prop(name, prop, float, None, ObserverPropertiesItemType.NUMBER, state, hint=hint)
        elif prop_type is bool:
            return self.prop(name, prop, parse_bool, None, ObserverPropertiesItemType.CHECKBOX, state, hint=hint)
        elif prop_type is List[int]:
            return self.prop(name, prop, self._parse_list_int, self._format_list, ObserverPropertiesItemType.TEXT,
                             state, hint=hint)
        elif prop_type is List[float]:
            return self.prop(name, prop, self._parse_list_float, self._format_list, ObserverPropertiesItemType.TEXT,
                             state, hint=hint)
        elif prop_type is Optional[List[int]]:
            return self.prop(name, prop, lambda v: None if v is None else self._parse_list_int(v),
                             lambda v: None if v is None else self._format_list(v), ObserverPropertiesItemType.TEXT,
                             state, optional=True, hint=hint)
        elif isinstance(prop_type, EnumMeta):
            select_items = list(prop_type)
            return self.prop(name, prop, lambda v: select_items[int(v)], lambda v: str(select_items.index(v)),
                             ObserverPropertiesItemType.SELECT, state,
                             select_values=[ObserverPropertiesItemSelectValueItem(e.name) for e in select_items],
                             hint=hint)
        elif is_tuple_type(prop_type):
            return make_tuple_prop(prop_type)
        else:
            raise IllegalArgumentException(f'Unrecognized prop type {prop_type}')

    def _check_instance_is_set(self):
        if self._instance is None:
            raise IllegalArgumentException(f'Instance not set. Pass `instance` param to __init__().')

    def prop(self,
             name: str,
             prop: property,
             parser: Optional[Callable[[str], T]],
             formatter: Optional[Callable[[T], str]],
             prop_type: ObserverPropertiesItemType = ObserverPropertiesItemType.TEXT,
             state: Optional[ObserverPropertiesItemState] = None,
             enabled: Optional[bool] = None,
             resolve_strategy: Optional[EditStrategy] = None,
             optional: bool = False,
             select_values: List[ObserverPropertiesItemSelectValueItem] = (),
             hint: str = ''):
        self._check_instance_is_set()
        state = self._resolve_state(state, enabled)
        value = prop.__get__(self._instance)
        if prop.fset is None:
            # Setter is not set -> item is marked as read_only
            if state == ObserverPropertiesItemState.ENABLED:
                state = ObserverPropertiesItemState.READ_ONLY
            setter = lambda _: None
        else:
            setter = partial(prop.__set__, self._instance)
        state, state_reason = self._resolve_state_strategy(state, resolve_strategy)
        return ObserverPropertiesItem.create(name, prop_type,
                                             formatter(value) if formatter is not None else value,
                                             self._create_update_value(setter, parser),
                                             state, optional,
                                             select_values=select_values,
                                             source_type=self._source_type,
                                             hint=hint)

    def button(self, name: str, action: Callable[[], None], caption: Optional[str] = None,
               state: ObserverPropertiesItemState = ObserverPropertiesItemState.ENABLED, hint: str = ''):
        def callback(_: str) -> str:
            action()
            return ''

        return ObserverPropertiesItem.create(name, ObserverPropertiesItemType.BUTTON,
                                             name if caption is None else caption, callback, state,
                                             source_type=self._source_type, hint=hint)

    @staticmethod
    def _resolve_state(state: Optional[ObserverPropertiesItemState],
                       enabled: Optional[bool]) -> ObserverPropertiesItemState:
        if enabled is not None and state is not None:
            raise IllegalArgumentException('Both arguments state and enabled cannot be set simultaneously. Set either '
                                           'one to None.')
        if state is not None:
            return state
        elif enabled is not None:
            return ObserverPropertiesItemState.ENABLED if enabled else ObserverPropertiesItemState.DISABLED
        else:
            return ObserverPropertiesItemState.ENABLED

    def _resolve_state_strategy(self, state: ObserverPropertiesItemState,
                                resolve_strategy: Optional[EditStrategy]
                                ) -> (ObserverPropertiesItemState, Optional[str]):
        if resolve_strategy is None:
            return state, None
        result = resolve_strategy(self._instance)
        if result is None:
            return state, None
        return result.state, result.reason
