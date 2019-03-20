
import pytest

from torchsim.gui.observer_system import Observable
from torchsim.gui.observer_view import ObserverView
from unittest.mock import MagicMock


@pytest.mark.parametrize("name, expected_name", [
    ('pref.header.one', 'one'),
    ('header.two', 'two'),
    ('name.header.pref.', 'header.pref.'),
])
def test_strip_observer_name_prefix(mocker, name: str, expected_name: str):
    observer_system: MagicMock = mocker.Mock()
    observer_view = ObserverView("test view", observer_system, "pref.")
    observer_view.set_observables({
        name: Observable(),
    })
    names = list(map(lambda i: i.name, observer_view.get_properties()))
    assert expected_name in names
