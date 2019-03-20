from torchsim.core.eval2.measurement_manager import RunMeasurementManager, MeasurementManager, MeasurementManagerParams


def get_something():
    return 42


def get_odd():
    def odd():
        odd.value = not odd.value
        return odd.value
    odd.value = True

    return odd


def test_measurement_functions():
    run_manager = RunMeasurementManager('foo topology', {})

    run_manager.add_measurement_f('foo 1', get_something)
    run_manager.add_measurement_f('foo 2', get_something, period=2)
    run_manager.add_measurement_f('foo odd', get_something, predicate=get_odd())
    run_manager.add_measurement_f_once('foo once', get_something, step=2)

    for step in range(0, 3):
        run_manager.step(step)

    measurements = run_manager.measurements

    assert [(0, 0), (1, 1), (2, 2)] == measurements.get_step_item_tuples('current_step')
    assert [(0, 42), (1, 42), (2, 42)] == measurements.get_step_item_tuples('foo 1')
    assert [(0, 42), (2, 42)] == measurements.get_step_item_tuples('foo 2')
    assert [(2, 42)] == measurements.get_step_item_tuples('foo once')
    assert [(1, 42)] == measurements.get_step_item_tuples('foo odd')


def test_measurement_manager():
    manager = MeasurementManager(None, MeasurementManagerParams())

    run = manager.create_new_run('foo model', {'param': 'value'})
    run._measurements.add(1, 'measurement', 42)

    manager.add_results(run)

    assert 42 == manager.single_run_measurements[0].get_item('measurement', step=1)
