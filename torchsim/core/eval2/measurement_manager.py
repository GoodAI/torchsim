import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Optional

from torchsim.core.eval2.run_measurement import SingleRunMeasurements

logger = logging.getLogger(__name__)


class CacheException(Exception):
    pass


class CacheNotFoundException(CacheException):
    pass


class RunMeasurementManager:
    """A measurement manager for a single experiment run."""

    def __init__(self, topology_name: str, topology_params: Dict[str, Any], cache_folder: str = None,
                 zip_files: bool = False, data_file: str = None):
        self._topology_name = topology_name
        self._topology_params = topology_params
        self.cache_folder = cache_folder
        self._use_zip = zip_files
        self._data_file = data_file
        self._measurements = SingleRunMeasurements(topology_name, topology_params, zip_files)
        self._measurement_functions = []

    @property
    def is_cached(self):
        return self._data_file is not None

    @property
    def measurements(self) -> SingleRunMeasurements:
        return self._measurements

    def load_from_cache(self):
        if self._data_file is None or not os.path.exists(self._data_file):
            raise CacheNotFoundException(self._data_file)

        self._measurements = SingleRunMeasurements.load_from_data_file(self._data_file)

    def save_cache(self):
        if not self.cache_folder:
            raise RuntimeError("No cache folder specified, cannot save")

        if not os.path.isdir(self.cache_folder):
            os.makedirs(self.cache_folder)

        if self.cache_folder is None:
            raise CacheException("Cache folder not specified")

        self._measurements.serialize_to(self.cache_folder)
        logger.info(f"Stored run results in {self.cache_folder}")

    def add_measurement_f(self, item_name: str, m_function: Any, period: int = 1,
                          predicate: Callable[[], bool] = None):
        self._measurement_functions.append((item_name, m_function, self.get_periodic_function(period, predicate)))

    def add_measurement_f_custom(self, item_name: str, m_function: Any, custom_f: Callable[[int], bool] = None):
        self._measurement_functions.append((item_name, m_function, custom_f))

    def add_measurement_f_once(self, item_name: str, m_function: Any, step: int = 0):
        self._measurement_functions.append((item_name, m_function, self.get_one_time_function(step)))

    def step(self, step: int):
        for name, measurement_function, permission_function in self._measurement_functions:
            if permission_function(step):
                self._measurements.add(step, name, measurement_function())

    @staticmethod
    def get_periodic_function(period, predicate):
        def periodic_f(step):
            # Either there is no predicate, or there is one and it's results is True.
            predicate_result = predicate is None or predicate()
            return step % period == 0 and predicate_result

        return periodic_f

    @staticmethod
    def get_one_time_function(step: int):
        def one_time_f(current_step):
            return current_step == step

        return one_time_f


@dataclass
class MeasurementManagerParams:
    delete_after_each_run: bool = True
    zip_data_files: bool = False


class MeasurementManager:
    """Stowage for measurements. Add values to last measurement each step."""
    single_run_measurements: List[SingleRunMeasurements]

    def __init__(self, cache_folder: Optional[str], params: Optional[MeasurementManagerParams]):
        params = params or MeasurementManagerParams()

        self.zip_data_files = params.zip_data_files
        self._cache_folder = cache_folder
        self.delete_after_each_run = params.delete_after_each_run
        self.single_run_measurements = []
        self._index_cache = None
        self._cached_runs = {}

    def _cache_folder_exists(self):
        return self._cache_folder and os.path.isdir(self._cache_folder)

    def clear_cache(self):
        if not self._cache_folder_exists():
            return

        for file_name in os.listdir(self._cache_folder):
            if file_name.endswith('.drm') or file_name.endswith('.crm'):
                os.unlink(os.path.join(self._cache_folder, file_name))

    def load_cache_index(self):
        if not self._cache_folder_exists():
            logger.warning(f"Cache folder not found during loading: {self._cache_folder}")
            return

        index_cache = SingleRunMeasurements.browse_content_files(self._cache_folder)

        for (model_name, params_string), description in index_cache.items():
            parameters = description['model_parameters']
            data_file = description['data_file']
            self._cached_runs[(model_name, params_string)] = RunMeasurementManager(
                model_name, parameters, self._cache_folder, self.zip_data_files, data_file)

    def create_new_run(self, model_name: str, model_params: Dict[str, Any]) -> RunMeasurementManager:
        return RunMeasurementManager(model_name, model_params, self._cache_folder, zip_files=self.zip_data_files)

    def add_results(self, single_run_manager: RunMeasurementManager):
        self.single_run_measurements.append(single_run_manager.measurements)

    def try_get_cached(self, model_name: str, model_params: Dict[str, Any]) -> Optional[RunMeasurementManager]:
        return self._cached_runs.get((model_name, str(SingleRunMeasurements.to_serializable_params(model_params))),
                                     None)

    def get_values_from_all_runs(self, item_name):
        return [single_run.get_items(item_name) for single_run in self.single_run_measurements]

    def get_custom_data_from_all_runs(self, item_name):
        return [single_run.get_custom_data(item_name) for single_run in self.single_run_measurements]

    def get_items_from_all_runs(self, item_name):
        return [single_run.get_step_item_tuples(item_name) for single_run in self.single_run_measurements]
