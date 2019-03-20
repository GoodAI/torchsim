import pytest
from functools import partial
from pytest import raises
from typing import List, Tuple, NamedTuple
from unittest.mock import MagicMock, call

from torchsim.core.exceptions import IllegalArgumentException
from torchsim.core.utils.singals import Signal0, Signal1, Signal2, Signal3, Signal4, Signal5, Signal6, signal


class Data(NamedTuple):
    a: int


class TestSignals:
    def test_signal_over_limit_raises(self):
        with raises(IllegalArgumentException, match="Signals up to [0-9]+ parameters are supported"):
            signal(int, int, int, int, int, int, int)

    @pytest.mark.parametrize('s', [
        Signal0(),
        signal()
    ])
    def test_signal_0(self, mocker, s):
        spy1: MagicMock = mocker.stub()
        spy2: MagicMock = mocker.stub()
        s.connect(spy1)
        s.connect(spy2)
        s.emit()
        spy1.assert_called_once()
        spy2.assert_called_once()

    @pytest.mark.parametrize('s', [
        Signal1(int),
        signal(int)
    ])
    def test_signal_1(self, mocker, s):
        spy1: MagicMock = mocker.stub()
        spy2: MagicMock = mocker.stub()
        s.connect(spy1)
        s.emit(5)
        s.connect(spy2)
        s.emit(10)
        spy1.assert_has_calls([
            call(5), call(10)
        ])
        spy2.assert_has_calls([
            call(10)
        ])

    @pytest.mark.parametrize('s', [
        Signal2(int, float),
        signal(int, float)
    ])
    def test_signal_2(self, mocker, s):
        spy1: MagicMock = mocker.stub()
        spy2: MagicMock = mocker.stub()
        s.connect(spy1)
        s.emit(5, 2.0)
        s.connect(spy2)
        s.emit(10, 1.0)
        spy1.assert_has_calls([
            call(5, 2.0), call(10, 1.0)
        ])
        spy2.assert_has_calls([
            call(10, 1.0)
        ])

    @pytest.mark.parametrize('s', [
        Signal3(int, float, str),
        signal(int, float, str)
    ])
    def test_signal_3(self, mocker, s):
        spy1: MagicMock = mocker.stub()
        spy2: MagicMock = mocker.stub()
        s.connect(spy1)
        s.connect(spy2)
        s.emit(5, 2.0, 'abc')
        spy1.assert_called_once_with(5, 2.0, 'abc')
        spy2.assert_called_once_with(5, 2.0, 'abc')

    @pytest.mark.parametrize('s', [
        Signal4(int, float, str, Tuple),
        signal(int, float, str, Tuple)
    ])
    def test_signal_4(self, mocker, s):
        spy: MagicMock = mocker.stub()
        s.connect(spy)
        s.emit(5, 2.0, 'abc', (1, 2, 3))
        spy.assert_called_once_with(5, 2.0, 'abc', (1, 2, 3))

    @pytest.mark.parametrize('s', [
        Signal5(int, float, str, Tuple, List),
        signal(int, float, str, Tuple, List)
    ])
    def test_signal_5(self, mocker, s):
        spy: MagicMock = mocker.stub()
        s.connect(spy)
        s.emit(5, 2.0, 'abc', (1, 2, 3), [0, 1])
        spy.assert_called_once_with(5, 2.0, 'abc', (1, 2, 3), [0, 1])

    @pytest.mark.parametrize('s', [
        Signal6(int, float, str, Tuple, List, Data),
        signal(int, float, str, Tuple, List, Data)
    ])
    def test_signal_6(self, mocker, s):
        spy: MagicMock = mocker.stub()
        s.connect(spy)
        s.emit(5, 2.0, 'abc', (1, 2, 3), [0, 1], Data(77))
        spy.assert_called_once_with(5, 2.0, 'abc', (1, 2, 3), [0, 1], Data(77))

    def test_callback_order(self):
        result = []

        def cb(value):
            result.append(value)

        s = signal()
        s.connect(partial(cb, 1))
        s.connect(partial(cb, 2))
        s.connect(partial(cb, 3))
        s.connect(partial(cb, 4))
        s.connect(partial(cb, 5))
        s.emit()
        assert [1, 2, 3, 4, 5] == result

    def test_connection_disable(self):
        result = []

        def cb(value):
            result.append(value)

        s = signal(int)
        connection = s.connect(cb)
        s.emit(1)
        connection.enabled = False
        s.emit(2)
        connection.enabled = True
        s.emit(3)
        assert [1, 3] == result

    def test_connection_disconnect(self):
        result = []

        def cb(prefix, value):
            result.append(f'{prefix}{value}')

        s = signal(int)
        c_a = s.connect(partial(cb, 'a'))
        c_b = s.connect(partial(cb, 'b'))
        s.emit(1)
        c_a.disconnect()
        s.emit(2)
        assert ['a1', 'b1', 'b2'] == result
