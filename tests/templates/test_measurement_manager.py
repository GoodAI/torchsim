from typing import Union, Tuple, Any, Dict, List

from torchsim.core.eval.experiment_template_base import TestableExperimentTemplateBase
from torchsim.core.eval.measurement_manager import MeasurementManagerBase
from torchsim.core.eval.testable_measurement_manager import TestableRunMeasurement
from torchsim.topologies.random_number_topology import RandomNumberTopology
from tests.templates.random_number_topology_adapter import RandomNumberTopologyTrainTestAdapter


class SimpleTestableExperimentTemplate(TestableExperimentTemplateBase):

    def __init__(self,
                 topology_adapter: RandomNumberTopologyTrainTestAdapter,
                 topology_class,
                 models_params: List[Union[Tuple[Any], Dict[str, Any]]],
                 overall_training_steps: int,
                 num_testing_steps: int,
                 num_testing_phases: int,
                 save_cache=False,
                 load_cache=False,
                 computation_only=False,
                 seed=None,
                 disable_plt_show=False):
        super().__init__(topology_adapter,
                         topology_class,
                         models_params,
                         overall_training_steps,
                         num_testing_steps,
                         num_testing_phases,
                         save_cache,
                         load_cache,
                         computation_only,
                         seed,
                         disable_plt_show,
                         clear_cache=False)

        self._measurement_manager = self._create_measurement_manager(self._experiment_folder,
                                                                     delete_after_each_run=False)

        self._measurement_manager.add_measurement_f_with_period_testing('testing_id',
                                                                        topology_adapter.get_output_id,
                                                                        1)

        self._measurement_manager.add_measurement_f_with_period_training('training_id',
                                                                         topology_adapter.get_output_id,
                                                                         1)

    def _get_measurement_manager(self) -> MeasurementManagerBase:
        return self._measurement_manager

    def _after_run_finished(self):
        pass

    def _compute_experiment_statistics(self):
        pass

    def _publish_results(self):
        pass

    def _experiment_template_name(self):
        return "testing_experiment_template"


def test_train_test_split():
    overall_training = 20
    num_testing_steps = 10
    num_testing_phases = 3

    topology_params = [{}, {}]

    template = SimpleTestableExperimentTemplate(
        RandomNumberTopologyTrainTestAdapter(),
        RandomNumberTopology,
        topology_params,
        overall_training_steps=overall_training,
        num_testing_steps=num_testing_steps,
        num_testing_phases=num_testing_phases,
        seed=1234,
        disable_plt_show=True)

    template.run()

    manager = template._get_measurement_manager()

    assert len(manager.run_measurements) == len(topology_params)
    assert manager.run_measurements[0].get_items_count('current_step') == \
        overall_training + num_testing_steps * num_testing_phases
    assert manager.run_measurements[0].get_items_count('testing_id') == num_testing_steps * num_testing_phases
    assert manager.run_measurements[0].get_items_count('training_id') == overall_training

    num_training_steps = overall_training // num_testing_phases
    accounted_for_steps = overall_training + num_testing_phases * num_testing_steps

    expected_testing_steps = []
    expected_testing_phase_ids = []
    for phase in range(num_testing_phases):
        steps = list(range(phase * num_testing_steps, (phase + 1) * num_testing_steps))
        expected_testing_steps.extend([-1] * num_training_steps + steps)
        expected_testing_phase_ids.extend([-1] * num_training_steps + [phase] * num_testing_steps)
    overflow = accounted_for_steps - len(expected_testing_steps)
    expected_testing_steps += overflow * [-1]
    expected_testing_phase_ids += overflow * [-1]

    expected_training_steps = []
    expected_training_phase_ids = []
    for phase in range(num_testing_phases):
        steps = list(range(phase * num_training_steps, (phase + 1) * num_training_steps))
        expected_training_steps.extend(steps + [-1] * num_testing_steps)
        expected_training_phase_ids.extend(([phase] * num_training_steps + [-1] * num_testing_steps))
    expected_training_steps += range(num_training_steps * num_testing_phases, overall_training)
    expected_training_phase_ids += [num_testing_phases] * overflow

    assert expected_testing_steps == manager.run_measurements[0].get_items('testing_step')
    assert expected_training_steps == manager.run_measurements[0].get_items('training_step')
    assert expected_testing_phase_ids == manager.run_measurements[0].get_items('testing_phase_id')
    assert expected_training_phase_ids == manager.run_measurements[0].get_items('training_phase_id')


def test_partition_to_phases():
    # values: dictionary: [key=global_sim_step: value=measured_value]
    values = {1: 'a', 2: 'b', 10: 'c', 11: 'd', 12: 'e', 13: 'f'}
    # ids: tuples: (global_sim_step, measured_phase), e.g. testing phase, -1 means no testing has been in that step
    ids = [(1, 5), (2, 5), (10, 5), (11, 6), (12, -1), (14, -1)]

    # create the run_measurement and write values there
    run_measurement = TestableRunMeasurement('model_name', zip_files=False)
    run_measurement._values['values'] = values
    run_measurement._values['training_phase_id'] = ids

    result = run_measurement.partition_to_training_phases('values')
    result_b = run_measurement.partition_to_phases('values', 'training_phase_id')
    # result is list of tuples, each tuple for each phase is: (phase [(step, value), (step, value)..])
    assert result == [(5, [(1, 'a'), (2, 'b'), (10, 'c')]), (6, [(11, 'd')])]
    assert result == result_b

    # better formatting: redundant phase id removed
    result = run_measurement.partition_to_list_of_training_phases('values', remove_steps=False)
    result_b = run_measurement.partition_to_list_of_phases('values', 'training_phase_id', remove_steps=False)
    assert result == [[(1, 'a'), (2, 'b'), (10, 'c')], [(11, 'd')]]
    assert result == result_b

    # steps removed
    result = run_measurement.partition_to_list_of_training_phases('values', remove_steps=True)
    result_b = run_measurement.partition_to_list_of_phases('values', 'training_phase_id', remove_steps=True)
    assert result == [['a', 'b', 'c'], ['d']]
    assert result == result_b

    # remove the training_phase_id, write it to the testing_phase_id, read the testing data
    run_measurement._values['training_phase_id'] = None
    run_measurement._values['testing_phase_id'] = ids

    result = run_measurement.partition_to_list_of_testing_phases('values', remove_steps=True)
    assert result == result_b




