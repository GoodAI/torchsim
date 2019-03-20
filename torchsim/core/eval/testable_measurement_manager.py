from collections import OrderedDict

from typing import Callable, List, Union, Any, Tuple

from torchsim.core.eval.measurement_manager import MeasurementManager
from torchsim.core.eval.run_measurement import RunMeasurement


class TestableRunMeasurement(RunMeasurement):

    def partition_to_training_phases(self, item_name: str):
        return self.partition_to_phases(item_name, 'training_phase_id')

    def partition_to_testing_phases(self, item_name: str):
        return self.partition_to_phases(item_name, 'testing_phase_id')

    def partition_to_phases(self, item_name: str, phase_source: Union[str, List[Tuple[int, int]]]='testing_phase_id') \
            -> List[Tuple[int, List[Tuple[int, Any]]]]:
        """Partition the values list to lists according to the same id.

        Args:
            item_name: name of the items to be partitioned
            phase_source: either string with name of item which serve as id, or list of tuples of step and id

        Returns:
            Partitioned lists of tuples (index, List[(tuple of step and value)]).
        """
        values = self.get_step_item_dict(item_name)
        if type(phase_source) == str:
            ids = self.get_step_item_tuples(phase_source)
        else:
            ids = phase_source

        results = OrderedDict()
        # go through the ids (ordered dict of measurement_id: (step, phase_id)
        for step, idx in ids:
            # ignore if the phase is not the one requested or the step is not in the values (not measured)
            if idx == -1 or step not in values:
                continue
            if idx not in results:
                results[idx] = []
            results[idx].append((step, values[step]))

        return list(results.items())

    def partition_to_list_of_training_phases(self, item_name: str, remove_steps: bool = True)\
            -> Union[List[List[Tuple[int, Any]]], List[List[Any]]]:
        """
        Partition a measurement given by name to training phases
        Args:
            item_name: name of the measured variable
            remove_steps: if true, the method will return List[List[Any]]: for each phase list of measured values
        Returns: measured values partitioned into training phases
        """
        return self.partition_to_list_of_phases(item_name, phase_source='training_phase_id', remove_steps=remove_steps)

    def partition_to_list_of_testing_phases(self, item_name: str, remove_steps: bool = True)\
            -> Union[List[List[Tuple[int, Any]]], List[List[Any]]]:
        """
        Partition a measurement given by name to testing phases
        Args:
            item_name: name of the measured variable
            remove_steps: if true, the method will return List[List[Any]]: for each phase list of measured values
        Returns: measured values partitioned into testing phases
        """
        return self.partition_to_list_of_phases(item_name, phase_source='testing_phase_id', remove_steps=remove_steps)

    def partition_to_list_of_phases(self, item_name: str,
                                    phase_source: Union[str, List[Tuple[int, int]]], remove_steps: bool = True) \
            -> Union[List[List[Tuple[int, Any]]], List[List[Any]]]:
        """
        Computes the same as the _partition_to_phases, but removes the redundant partition id
        Args:
            remove_steps: optionally can remove steps from the resulting list of phases
            item_name: name of the items to be partitioned
            phase_source: either string with name of item which serve as id, or list of tuples of step and id
        Returns: list of partitions, for each partition there is a list of measured values
        """
        list_tuples = self.partition_to_phases(item_name, phase_source)
        phases = []
        for phase in list_tuples:
            phases.append(phase[1])

        if remove_steps:
            return TestableRunMeasurement._remove_steps(phases)
        return phases

    @staticmethod
    def concatenate_lists_of_phases(phases: List[List[Any]]) -> List[Any]:
        output = [x for n in phases for x in n]
        return output

    @staticmethod
    def _remove_steps(list_tuples: List[List[Tuple[int, Any]]]) -> List[List[Any]]:
        """
        Remove the steps from a given list of tuples
        Args:
            list_tuples:
        Returns: List (list of phases) of lists (list of measured values)

        """
        phases = [[x[1] for x in phase] for phase in list_tuples]

        return phases


class TestableMeasurementManager(MeasurementManager):
    _is_training_function: Callable
    _is_testing_function: Callable
    run_measurements: List[Union[TestableRunMeasurement, None]]
    run_measurement_class = TestableRunMeasurement

    def __init__(self,
                 is_training_function: Callable,
                 testing_phase_id_f: Callable,
                 training_phase_id_f: Callable,
                 testing_step_f: Callable,
                 training_step_f: Callable,
                 is_testing_function: Callable = None,
                 cache_folder=None,
                 delete_after_each_run=False,
                 zip_data_files=False):
        super().__init__(cache_folder, delete_after_each_run, zip_data_files)

        self._is_training_function = is_training_function
        if is_testing_function is None:
            def is_testing_function(**kwargs):
                return not is_training_function(**kwargs)

        self._is_testing_function = is_testing_function

        self.add_measurement_f_with_period('testing_phase_id', testing_phase_id_f, 1)
        self.add_measurement_f_with_period('training_phase_id', training_phase_id_f, 1)
        self.add_measurement_f_with_period('testing_step', testing_step_f, 1)
        self.add_measurement_f_with_period('training_step', training_step_f, 1)


    def add_measurement_f_with_period_training(self,
                                               item_name,
                                               m_function: Callable,
                                               period: int):
        """Runs the measurement with given period only during training."""
        super().add_measurement_f_with_period_and_custom_permission_f(item_name,
                                                                      m_function,
                                                                      self._is_training_function,
                                                                      period)

    def add_measurement_f_with_period_testing(self,
                                              item_name,
                                              m_function: Callable,
                                              period: int):
        """Runs the measurement with given period only during testing."""
        super().add_measurement_f_with_period_and_custom_permission_f(item_name,
                                                                      m_function,
                                                                      self._is_testing_function,
                                                                      period)
