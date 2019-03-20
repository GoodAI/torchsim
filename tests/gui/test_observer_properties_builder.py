import pytest
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pytest import raises
from typing import List, Optional, Tuple

from torchsim.core.exceptions import IllegalArgumentException, FailedValidationException
from torchsim.gui.observables import ObserverPropertiesBuilder, ObserverPropertiesItemState, ObserverPropertiesItemType, \
    Initializable, enable_on_runtime, disable_on_runtime
from torchsim.gui.validators import *


class DataIsInitializable(Initializable):
    _initializable: bool

    def __init__(self, initializable: bool):
        self._initializable = initializable

    def is_initialized(self) -> bool:
        return self._initializable


class DataIsNotInitializable:
    pass


class Types(Enum):
    ONE = 1,
    TWO = 2,
    THREE = 3

class Data:
    _int: int
    _float: float
    _bool: bool
    _int_list: List[int]
    _optional_int: Optional[int]
    _optional_list_int: Optional[List[int]]
    _enum: Types
    _str: str
    _tuple_int_int: Tuple[int, int]
    _tuple_float_float: Tuple[float, float]

    @property
    def p_int(self) -> int:
        return self._int

    @p_int.setter
    def p_int(self, value):
        self._int = value

    @property
    def p_int_no_type(self):
        return self._int

    @property
    def p_float(self) -> float:
        return self._float

    @p_float.setter
    def p_float(self, value):
        self._float = value

    @property
    def p_float_no_setter(self) -> float:
        return self._float

    @property
    def p_bool(self) -> bool:
        return self._bool

    @p_bool.setter
    def p_bool(self, value):
        self._bool = value

    @property
    def p_int_list(self) -> List[int]:
        return self._int_list

    @p_int_list.setter
    def p_int_list(self, value):
        self._int_list = value

    @property
    def p_optional_int(self) -> Optional[int]:
        return self._optional_int

    @p_optional_int.setter
    def p_optional_int(self, value):
        self._optional_int = value

    @property
    def p_optional_list_int(self) -> Optional[List[int]]:
        return self._optional_list_int

    @p_optional_list_int.setter
    def p_optional_list_int(self, value):
        self._optional_list_int = value

    @property
    def p_enum(self) -> Types:
        return self._enum

    @p_enum.setter
    def p_enum(self, value):
        self._enum = value

    @property
    def p_str(self) -> str:
        return self._str

    @p_str.setter
    def p_str(self, value):
        self._str = value

    @property
    def p_tuple_int_int(self) -> Tuple[int, int]:
        return self._tuple_int_int

    @p_tuple_int_int.setter
    def p_tuple_int_int(self, value):
        self._tuple_int_int = value

    @property
    def p_tuple_float_float(self) -> Tuple[float, float]:
        return self._tuple_float_float

    @p_tuple_float_float.setter
    def p_tuple_float_float(self, value):
        self._tuple_float_float = value


class TestObserverPropertiesBuilder(ABC):
    builder: ObserverPropertiesBuilder

    def setup_method(self):
        self.data = Data()
        self.builder = ObserverPropertiesBuilder(self.data)

    @pytest.mark.parametrize("input,expected", [
        ([], ''),
        ([1], '1'),
        ([1, 2, 3], '1,2,3'),
        ([1.5, 2, 'abc'], '1.5,2,abc'),
    ])
    def test_format_list(self, input, expected):
        assert expected == ObserverPropertiesBuilder._format_list(input)

    @pytest.mark.parametrize("input,expected", [
        ('', []),
        ('1', [1]),
        ('1,2,3', [1, 2, 3]),
    ])
    def test_parse_list(self, input, expected):
        assert expected == ObserverPropertiesBuilder._parse_list(input, int)

    @pytest.mark.parametrize("input,expected", [
        ([], ''),
        ([1], '<1>'),
        ([1, 2, 3], '<1>,<2>,<3>'),
    ])
    def test_format_list(self, input, expected):
        assert expected == ObserverPropertiesBuilder._format_list(input, lambda i: f'<{i}>')

    def test_auto_needs_instance_to_be_set(self):
        with raises(IllegalArgumentException, match=r'.*Instance not set.*'):
            builder = ObserverPropertiesBuilder()
            builder.auto("Test", Data.p_int)

    def test_auto_undefined_type(self):
        with raises(IllegalArgumentException, match=r'.*Property getter must be annotated*'):
            self.builder.auto("Test", Data.p_int_no_type)

    def test_auto_no_setter_means_readonly(self):
        self.data.p_float = 1.0
        item = self.builder.auto("Test", Data.p_float_no_setter)
        assert 1.0 == item.value
        assert ObserverPropertiesItemState.READ_ONLY == item.state
        item.callback('1.5')
        assert 1.0 == item.value

    def test_auto_int(self):
        self.data.p_int = 10
        item = self.builder.auto("Test", Data.p_int)
        assert 10 == item.value
        assert ObserverPropertiesItemType.NUMBER == item.type
        assert ObserverPropertiesItemState.ENABLED == item.state
        item.callback("20")
        assert 20 == self.data.p_int

    def test_auto_str(self):
        self.data.p_str = 'abc'
        item = self.builder.auto("Test", Data.p_str)
        assert 'abc' == item.value
        assert ObserverPropertiesItemType.TEXT == item.type
        assert ObserverPropertiesItemState.ENABLED == item.state
        item.callback("text")
        assert "text" == self.data.p_str

    def test_auto_float(self):
        self.data.p_float = -1.4
        item = self.builder.auto("Test", Data.p_float)
        assert -1.4 == item.value
        item.callback("-2.14")
        assert ObserverPropertiesItemType.NUMBER == item.type
        assert ObserverPropertiesItemState.ENABLED == item.state
        assert -2.14 == self.data.p_float

    def test_auto_bool(self):
        self.data.p_bool = True
        item = self.builder.auto("Test", Data.p_bool)
        assert True is item.value
        assert ObserverPropertiesItemType.CHECKBOX == item.type
        item.callback("False")
        assert False is self.data.p_bool

    def test_auto_list_int(self):
        self.data.p_int_list = [1, 2, 3]
        item = self.builder.auto("Test", Data.p_int_list)
        assert '1,2,3' == item.value
        assert ObserverPropertiesItemType.TEXT == item.type
        item.callback("5,6")
        assert [5, 6] == self.data.p_int_list

    @pytest.mark.parametrize('input_value', [
        "5,1.2,5",
        "5,abc,5",
        "abc"
    ])
    def test_auto_list_int_invalid_input(self, input_value):
        with raises(FailedValidationException, match="Expected List\[int\], syntax error:"):
            self.data.p_int_list = [1, 2, 3]
            item = self.builder.auto("Test", Data.p_int_list)
            item.callback(input_value)

    def test_optional_int(self):
        self.data.p_optional_int = None
        item = self.builder.auto("Test", Data.p_optional_int)
        assert None is item.value
        assert ObserverPropertiesItemType.NUMBER == item.type
        assert True is item.optional
        item.callback("1")
        assert 1 == self.data.p_optional_int

    def test_optional_list_int(self):
        self.data.p_optional_list_int = None
        item = self.builder.auto("Test", Data.p_optional_list_int)
        assert None is item.value
        assert ObserverPropertiesItemType.TEXT == item.type
        assert True is item.optional
        item.callback("1,2")
        assert [1, 2] == self.data.p_optional_list_int

    def test_enum(self):
        self.data.p_enum = Types.ONE
        item = self.builder.auto("Test", Data.p_enum)
        assert '0' == item.value
        assert ObserverPropertiesItemType.SELECT == item.type
        item.callback('1')
        assert Types.TWO == self.data.p_enum

    def test_tuple_int_int(self):
        self.data.p_tuple_int_int = (10, 20)
        item = self.builder.auto("Test", Data.p_tuple_int_int)
        assert '10,20' == item.value
        assert ObserverPropertiesItemType.TEXT == item.type
        item.callback('1,2')
        assert (1, 2) == self.data.p_tuple_int_int

    @pytest.mark.parametrize("param, exception", [
        ('1', 'Expected exactly 2 items, but 1 received'),
        ('1, 2, 3', 'Expected exactly 2 items, but 3 received')
    ])
    def test_tuple_int_int(self, param, exception):
        self.data.p_tuple_int_int = (10, 20)
        item = self.builder.auto("Test", Data.p_tuple_int_int)
        with raises(FailedValidationException, match=exception):
            item.callback(param)

    def test_tuple_float_float(self):
        self.data.p_tuple_float_float = (1.1, 1.2)
        item = self.builder.auto("Test", Data.p_tuple_float_float)
        assert '1.1,1.2' == item.value
        assert ObserverPropertiesItemType.TEXT == item.type
        item.callback('3.5,4.7')
        assert (3.5,4.7) == self.data.p_tuple_float_float

    @pytest.mark.parametrize("param, exception", [
        ('1', 'Expected exactly 2 items, but 1 received'),
        ('1, 2, 3', 'Expected exactly 2 items, but 3 received')
    ])
    def test_tuple_float_float(self, param, exception):
        self.data.p_tuple_float_float = (1.1, 1.2)
        item = self.builder.auto("Test", Data.p_tuple_float_float)
        with raises(FailedValidationException, match=exception):
            item.callback(param)

    @pytest.mark.parametrize("state, enabled, result, should_pass", [
        (ObserverPropertiesItemState.ENABLED, None, ObserverPropertiesItemState.ENABLED, True),
        (ObserverPropertiesItemState.DISABLED, None, ObserverPropertiesItemState.DISABLED, True),
        (ObserverPropertiesItemState.READ_ONLY, None, ObserverPropertiesItemState.READ_ONLY, True),
        (None, True, ObserverPropertiesItemState.ENABLED, True),
        (None, False, ObserverPropertiesItemState.DISABLED, True),
        (None, None, ObserverPropertiesItemState.ENABLED, True),  # default is enabled
        (ObserverPropertiesItemState.ENABLED, False, None, False),
    ])
    def test_resolve_state(self, state, enabled, result, should_pass):
        if should_pass:
            assert result == ObserverPropertiesBuilder._resolve_state(state, enabled)
        else:
            with raises(IllegalArgumentException):
                ObserverPropertiesBuilder._resolve_state(state, enabled)

    @pytest.mark.parametrize("initializable, state, strategy, exp_state, exp_description", [
        (True, ObserverPropertiesItemState.ENABLED, enable_on_runtime, ObserverPropertiesItemState.ENABLED, None),
        (True, ObserverPropertiesItemState.DISABLED, enable_on_runtime, ObserverPropertiesItemState.DISABLED, None),
        (True, ObserverPropertiesItemState.READ_ONLY, enable_on_runtime, ObserverPropertiesItemState.READ_ONLY, None),
        (False, ObserverPropertiesItemState.ENABLED, enable_on_runtime, ObserverPropertiesItemState.DISABLED,
         "Item can be modified only when the simulation is running"),
        (False, ObserverPropertiesItemState.DISABLED, enable_on_runtime, ObserverPropertiesItemState.DISABLED,
         "Item can be modified only when the simulation is running"),
        (False, ObserverPropertiesItemState.ENABLED, disable_on_runtime, ObserverPropertiesItemState.ENABLED, None),
        (True, ObserverPropertiesItemState.ENABLED, disable_on_runtime, ObserverPropertiesItemState.DISABLED,
         "Item can be modified only when the simulation is stopped")
    ])
    def test_resolve_state_edit_strategy(self, initializable, state, strategy, exp_state, exp_description):
        builder = ObserverPropertiesBuilder(DataIsInitializable(initializable))
        res_state, res_description = builder._resolve_state_strategy(state, strategy)
        assert exp_state == res_state
        assert exp_description == res_description

    def test_resolve_state_edit_strategy_exception(self):
        builder = ObserverPropertiesBuilder(DataIsNotInitializable())
        with raises(IllegalArgumentException,
                    match=r'Expected instance of .*Initializable.* but .*DataIsNotInitializable.* received.'):
            builder._resolve_state_strategy(ObserverPropertiesItemState.ENABLED, enable_on_runtime)
