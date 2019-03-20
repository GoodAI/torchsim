from pytest import raises

from torchsim.core.exceptions import PrivateConstructorException
from torchsim.core.global_settings import GlobalSettings


def test_is_singleton():
    s1 = GlobalSettings.instance()
    s2 = GlobalSettings.instance()
    assert s1 == s2


def test_constructor_raises_exception():
    with raises(PrivateConstructorException):
        GlobalSettings()


def test_no_direct_access():
    with raises(AttributeError):
        _ = GlobalSettings.observer_memory_block_minimal_size


def test_instance_read_access():
    assert 50 == GlobalSettings.instance().observer_memory_block_minimal_size


def test_instance_write_access():
    GlobalSettings.instance().observer_memory_block_minimal_size = 20
    assert 20 == GlobalSettings.instance().observer_memory_block_minimal_size
