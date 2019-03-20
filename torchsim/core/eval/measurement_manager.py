from abc import abstractmethod
from typing import Dict, Any, List, Union, Callable

from torchsim.core.eval.run_measurement import RunMeasurement


class MeasurementManagerBase:
    """Stowage for measurements. Add values to last measurement each step."""
    run_measurements: List[Union[RunMeasurement, None]]
    run_measurement_class = RunMeasurement

    def __init__(self, cache_folder=None, delete_after_each_run=False, zip_data_files=False):
        self.zip_data_files = zip_data_files
        if cache_folder is not None:
            self._cache_folder = cache_folder
            self._cache_content = self.run_measurement_class.browse_content_files(cache_folder)
        self.delete_after_each_run = delete_after_each_run
        self.run_measurements = []

    @abstractmethod
    def step(self, step: int):
        raise NotImplementedError()

    def create_new_run(self, model_name: str, model_params: Dict[str, Any]):
        self.run_measurements.append(self.run_measurement_class(model_name, model_params, zip_files=self.zip_data_files))

    def cache_last_run(self):
        self.run_measurements[-1].serialize_to(self._cache_folder)

    def is_run_cached(self, model_name: str, model_params: Dict[str, Any]) -> bool:
        return (model_name, str(RunMeasurement.to_serializable_params(model_params))) in self._cache_content

    def try_load_run(self, model_name: str, model_params: Dict[str, Any]):
        if self.is_run_cached(model_name, model_params):
            drm_file = self._cache_content[(model_name, str(RunMeasurement.to_serializable_params(model_params)))]
            self.run_measurements.append(self.run_measurement_class.load_from_data_file(drm_file))
            return True
        return False

    def finish_run(self):
        if self.delete_after_each_run and len(self.run_measurements):
            self.run_measurements[-1] = None


class MeasurementManager(MeasurementManagerBase):
    """Universal implementation of MeasurementManagerBase.

    `m_function` is function measuring values.
    `permission_function` is telling whether the value should be measured at the step.

    Currently, only `step` is provided for permission function args.

    If you need more parameters in permission function, add them to TopologyAdapter and pass adapter directly
    (like partial function application):
    `
    ...
    adapter = SpecialAdapter()
    def special_permission_f(**kwargs):
        return adapter.should_measure()

    measurement_manager.add_measurement_f_with_custom_permission_f(item_name, m_function, special_permission_f)
    `
    """

    def __init__(self, cache_folder=None, delete_after_each_run=False, zip_data_files=False):
        super().__init__(cache_folder, delete_after_each_run, zip_data_files)
        self._measurement_functions = []

    def add_measurement_f(self, item_name: str, m_function: Any):
        self._measurement_functions.append((item_name, m_function, self.get_periodic_function(1)))

    def add_measurement_f_with_period(self, item_name: str, m_function: Any, period: int):
        self._measurement_functions.append((item_name, m_function, self.get_periodic_function(period)))

    def add_measurement_f_with_period_and_custom_permission_f(self, item_name: str,
                                                              m_function: Callable,
                                                              permission_function: Callable,
                                                              period: int):
        """Checks both conditions: periodic measurement and custom permission function."""
        self._measurement_functions.append((
            item_name,
            m_function,
            self.get_periodic_custom_function(period, permission_function)))

    def add_measurement_f_with_custom_permission_f(self, item_name: str, m_function: Any, permission_function: Any):
        self._measurement_functions.append((item_name, m_function, permission_function))

    def add_measurement_f_once(self, item_name: str, m_function: Any, step: int = 0):
        self._measurement_functions.append((item_name, m_function, self.get_one_time_function(step)))

    def step(self, step: int):
        for name, measurement_function, permission_function in self._measurement_functions:
            if permission_function(step=step):
                self.run_measurements[-1].add(step, name, measurement_function())

    @staticmethod
    def get_periodic_custom_function(period, custom_permission_f):
        def periodic_custom_f(**kwargs):
            return kwargs['step'] % period == 0 and custom_permission_f(**kwargs)
        return periodic_custom_f

    @staticmethod
    def get_periodic_function(period):
        def periodic_f(**kwargs):
            return kwargs['step'] % period == 0

        return periodic_f

    @staticmethod
    def get_one_time_function(step: int):
        def one_time_f(**kwargs):
            return kwargs['step'] == step

        return one_time_f
