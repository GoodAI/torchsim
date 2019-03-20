import pytest

from torchsim.core.eval.run_measurement import RunMeasurement


def test_names():
    rm = RunMeasurement('', {})

    rm.add(0, 'a', 'a val 1')
    rm.add(1, 'b', 'b val 1')
    rm.add(1, 'a', 'a val 2')
    rm.add(10, 'a', 'a val 3')
    rm.add(11, 'b', 'v val 2')

    names = rm.get_item_names()

    assert 'a' in names
    assert 'b' in names
    assert 'current_step' in names
    assert 'c' not in names


def test_get_item():
    rm = RunMeasurement('', {})

    rm.add(0, 'a', 'a val 1')
    rm.add(1, 'b', 'b val 1')
    rm.add(1, 'a', 'a val 2')
    rm.add(10, 'a', 'a val 3')
    rm.add(11, 'b', 'b val 2')

    assert rm.get_item('a', 0) == 'a val 1'
    assert rm.get_item('b', 1) == 'b val 1'
    assert rm.get_item('a', 1) == 'a val 2'
    assert rm.get_item('a', 10) == 'a val 3'
    assert rm.get_item('b', 11) == 'b val 2'


def test_get_step():
    rm = RunMeasurement('', {})

    rm.add(2, 'a', 'a val 1')
    rm.add(3, 'b', 'b val 1')
    rm.add(3, 'a', 'a val 2')
    rm.add(10, 'a', 'a val 3')
    rm.add(11, 'b', 'b val 2')

    assert rm.get_first_step() == 2
    assert rm.get_last_step() == 11


def test_get_count():
    rm = RunMeasurement('', {})

    rm.add(2, 'a', 'a val 1')
    rm.add(3, 'b', 'b val 1')
    rm.add(3, 'a', 'a val 2')
    rm.add(10, 'a', 'a val 3')
    rm.add(11, 'b', 'b val 2')

    assert rm.get_items_count('a') == 3
    assert rm.get_items_count('b') == 2


def test_dict():
    rm = RunMeasurement('', {})

    rm.add(2, 'a', 'a val 1')
    rm.add(3, 'b', 'b val 1')
    rm.add(3, 'a', 'a val 2')
    rm.add(10, 'a', 'a val 3')
    rm.add(11, 'b', 'b val 2')

    items_dict = rm.get_step_item_dict('a')
    assert items_dict[2] == 'a val 1'
    assert items_dict[3] == 'a val 2'
    assert items_dict[10] == 'a val 3'


def test_iterator():
    rm = RunMeasurement('', {})

    rm.add(2, 'a', 'a val 1')
    rm.add(3, 'b', 'b val 1')
    rm.add(3, 'a', 'a val 2')
    rm.add(10, 'a', 'a val 3')
    rm.add(11, 'b', 'b val 2')

    assert next(rm)['a'] == 'a val 1'
    assert next(rm)['b'] == 'b val 1'
    assert next(rm)['a'] == 'a val 3'
    with pytest.raises(KeyError):
        next(rm)['a'] == 'a val 3'


def test_items_list():
    rm = RunMeasurement('', {})

    rm.add(2, 'a', 'a val 1')
    rm.add(3, 'b', 'b val 1')
    rm.add(3, 'a', 'a val 2')
    rm.add(10, 'a', 'a val 3')
    rm.add(11, 'b', 'b val 2')

    items_list = rm.get_items('b')
    assert items_list[0] == 'b val 1'
    assert items_list[1] == 'b val 2'


def test_wrong_add_catch():
    rm = RunMeasurement('', {})

    rm.add(2, 'a', 'a val 1')
    rm.add(3, 'b', 'b val 1')
    rm.add(3, 'a', 'a val 2')
    with pytest.raises(ValueError):
        rm.add(2, 'a', 'a val 3')
