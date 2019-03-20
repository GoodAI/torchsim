import dacite
import pytest
from typing import List, Any

from unittest.mock import MagicMock, call

from torchsim.gui.server.ui_helper import UIHelper, PropertiesHolder
from torchsim.gui.observables import ObserverPropertiesItemSourceType
from torchsim.gui.observer_system import Observable, ObserverPropertiesItem, SimpleFileObserverPersistence
from torchsim.gui.server.ui_server_connector import EventData, EventDataPropertyUpdated


class ObservableStub(Observable):
    def __init__(self, spy):
        self.spy = spy

    def get_data(self) -> Any:
        return None

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._createItem("int", "number", 3),
            self._createItem("int_model", "number", 3, is_model=True),
            self._createItem("float", "number", 3.1),
            self._createItem("string", "text", "one"),
            self._createItem("bool_t", "checkbox", True),
            self._createItem("bool_f", "checkbox", False),
            self._createItem("button", "button", "click_me"),
        ]

    def _createItem(self, name: str, item_type: str, value, is_model: bool = False):
        return ObserverPropertiesItem(name, item_type, value, lambda new_value: self.spy(name, item_type, new_value),
                                      source_type=ObserverPropertiesItemSourceType.MODEL if is_model else ObserverPropertiesItemSourceType.OBSERVER)


class TestObserverPersistence:
    @pytest.fixture()
    def persistence(self, tmpdir, mocker):
        file = str(tmpdir.join("test.yaml"))
        persistence = SimpleFileObserverPersistence(file)
        yield persistence
        persistence.stop()

    def test_default_values(self, persistence, mocker):
        stub1: MagicMock = mocker.stub()
        o1 = ObservableStub(stub1)

        persistence.store_value("o1", "string", "abc")
        persistence.set_default_values("o1", o1)

        persistence.read_stored_values("o1", o1)
        stub1.assert_has_calls([
            call("int", "number", 3),
            call("float", "number", 3.1),
            call("bool_t", "checkbox", True),
            call("bool_f", "checkbox", False),
            call("string", "text", "abc")
        ], any_order=True)

    def test_store_load_in_memory(self, persistence, mocker):
        persistence.store_value("o1", "int", 5)
        persistence.store_value("o1", "float", 2.4)
        persistence.store_value("o1", "bool_t", False)
        persistence.store_value("o1", "bool_f", True)
        persistence.store_value("o1", "string", "abc")

        persistence.store_value("o2", "int", 10)

        stub1: MagicMock = mocker.stub()
        o1 = ObservableStub(stub1)
        persistence.read_stored_values("o1", o1)
        stub1.assert_has_calls([
            call("int", "number", 5),
            call("float", "number", 2.4),
            call("bool_t", "checkbox", False),
            call("bool_f", "checkbox", True),
            call("string", "text", "abc")
        ])

        stub2: MagicMock = mocker.stub()
        o2 = ObservableStub(stub2)
        persistence.read_stored_values("o2", o2)
        stub2.assert_has_calls([
            call("int", "number", 10)
        ])

    def test_store_load_from_file(self, tmpdir, mocker):
        file = str(tmpdir.join("test.yaml"))
        p_write = SimpleFileObserverPersistence(file)
        p_write.store_value("o1", "int", 5)
        p_write.store_value("o1", "float", 2.4)
        p_write.store_value("o1", "bool_t", False)
        p_write.store_value("o1", "bool_f", True)
        p_write.store_value("o1", "string", "abc")

        p_write.store_value("o2", "int", 10)

        p_write._write_to_file()
        p_write.stop()

        p_read = SimpleFileObserverPersistence(file)
        stub1: MagicMock = mocker.stub()
        o1 = ObservableStub(stub1)
        p_read.read_stored_values("o1", o1)
        stub1.assert_has_calls([
            call("int", "number", 5),
            call("float", "number", 2.4),
            call("bool_t", "checkbox", False),
            call("bool_f", "checkbox", True),
            call("string", "text", "abc")
        ])

        stub2: MagicMock = mocker.stub()
        o2 = ObservableStub(stub2)
        p_read.read_stored_values("o2", o2)
        stub2.assert_has_calls([
            call("int", "number", 10)
        ])
        p_read.stop()

    def test_store_load_ignores_buttons(self, persistence, mocker):
        persistence.store_value("o1", "int", 5)
        persistence.store_value("o1", "button", "click_me")

        stub1: MagicMock = mocker.stub()
        o1 = ObservableStub(stub1)
        persistence.read_stored_values("o1", o1)
        stub1.assert_has_calls([
            call("int", "number", 5),
            # call("button", "button", "click_me"), # should not be called
        ], any_order=True)
        assert stub1.call_count == 1

    def test_store_load_model_source_type_is_read_only_on_force_load(self, persistence, mocker):
        persistence.store_value("o1", "int", 5)
        persistence.store_value("o1", "int_model", 5)

        stub1: MagicMock = mocker.stub()
        o1 = ObservableStub(stub1)
        persistence.read_stored_values("o1", o1)
        stub1.assert_has_calls([
            call("int", "number", 5),
            # call("int_model", "number", 5),
        ], any_order=True)
        assert stub1.call_count == 1
        stub1.reset_mock()
        persistence.read_stored_values("o1", o1, True)
        stub1.assert_has_calls([
            call("int", "number", 5),
            call("int_model", "number", 5),
        ], any_order=True)
        assert stub1.call_count == 2


class TestObserverPersistencePropsAreStoredOnValueChange:

    @pytest.fixture()
    def observer_persistence(self, tmpdir, mocker):
        file = str(tmpdir.join("test.yaml"))
        observer_persistence = SimpleFileObserverPersistence(file)
        observer_persistence.store_value = mocker.stub()
        yield observer_persistence
        observer_persistence.stop()

    @pytest.fixture()
    def prop_callback(self, mocker, observer_persistence):
        connector: MagicMock = mocker.stub()
        helper = UIHelper(connector, observer_persistence)
        # prop_spy: MagicMock = mocker.stub()
        o1 = ObservableStub(lambda win, item_name, value: int(value))
        properties_holder = PropertiesHolder(o1.get_properties())
        prop_callback = helper.get_properties_callback('o1', properties_holder)
        return prop_callback

    def test_non_model_props_are_stored_on_value_change(self, observer_persistence, prop_callback):
        # noinspection PyArgumentList
        event = EventDataPropertyUpdated(win_id='o1', event_type='property_updated', property_id=0, value='10')
        prop_callback(event)
        observer_persistence.store_value.assert_called_once_with('o1', 'int', 10)
        del observer_persistence

    def test_model_props_are_not_stored_on_value_change(self, observer_persistence, prop_callback):
        # noinspection PyArgumentList
        event = EventDataPropertyUpdated(win_id='o1', event_type='property_updated', property_id=1, value='10')
        prop_callback(event)
        assert observer_persistence.store_value.call_count == 0
