from collections import OrderedDict

import glob
import gzip
import os
import pickle
import platform
from os import path
from ruamel import yaml
from ruamel.yaml import YAML
from typing import List, Any, Dict, Tuple, Union

from torchsim.core.eval.series_plotter import get_stamp, to_safe_path, to_safe_name


def save_zip(object_to_save, filename, protocol=-1):
    """Save an object to a compressed disk file.

       Works well with huge objects.
    """
    with gzip.GzipFile(filename, 'wb') as file:
        pickle.dump(object_to_save, file, protocol)


def load_zip(filename):
    """Loads a compressed object from disk."""
    filename = to_safe_path(filename)
    with gzip.GzipFile(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


class SingleRunMeasurements:
    """Class for easy collecting and retrieval of measured values.

    Designed to work with one run - when adding, `step` value must always increase.
    """
    model_name: str
    model_parameters: Dict[str, Any]

    def __init__(self, model_name: str = '', model_parameters: Dict[str, Any] = None, zip_files: bool = False):
        """Initialize with info about measured model.

        Args:
            model_name:
            model_parameters: dictionary of parameter names as keys and values
        """
        self.model_name = model_name
        self.model_parameters = model_parameters or {}
        self._values = {}
        self._dict = {}
        self._dict_change = True
        self._list_change = True
        self._steps = []
        self._values['current_step'] = []
        self._iter_counter = 0
        self.use_zip = zip_files
        self._custom_data = {}

    def add(self, step: int, item_name: str, value: Any):
        """Add a given new value and notice a step. Value can be later accessed via item_name and step.

        Calling with `step` param smaller than in previous calls raise ValueError.
        """
        self._dict_change = True
        if len(self._steps) > 0 and self._steps[-1] > step:
            raise ValueError("RunMeasurement class assumes add() is called with step parameter "
                             "larger or same as previous call.")
        if len(self._steps) == 0:
            self._steps.append(step)
        elif self._steps[-1] != step:
            self._steps.append(step)
        if item_name not in self._values:
            self._values[item_name] = []
        self._values[item_name].append((step, value))
        if (not len(self._values['current_step'])) or self._values['current_step'][-1] != (step, step):
            self._values['current_step'].append((step, step))

    def add_custom_data(self, item_name: str, values: Any):
        self._custom_data[item_name] = values

    def get_custom_data(self, item_name: str):
        return self._custom_data[item_name]

    def get_item_names(self) -> List[str]:
        return list(self._values.keys())

    def get_step_item_dict(self, item_name: str) -> OrderedDict:
        """Create a dictionary of steps -> measured items.

        Returns:
            OrderedDictionary, where keys are steps and values are measured items.
        """
        self._raise_item_name_error(item_name)
        if item_name not in self._dict or self._dict_change:
            self._dict[item_name] = OrderedDict(self._values[item_name])
            self._dict_change = False
        return self._dict[item_name]

    def get_item(self, item_name: str, step: int) -> Any:
        self._raise_item_name_error(item_name)
        return self.get_step_item_dict(item_name)[step]

    def get_items(self, item_name: str) -> List[Any]:
        self._raise_item_name_error(item_name)
        return [value for step, value in self._values[item_name]]

    def get_step_item_tuples(self, item_name: str) -> List[Tuple[int, int]]:
        self._raise_item_name_error(item_name)
        return self._values[item_name]

    def __iter__(self):
        self._iter_counter = 0
        return self

    def __next__(self):
        if self._iter_counter == len(self._steps):
            raise StopIteration()
        step = self._steps[self._iter_counter]
        self._iter_counter += 1
        return dict(
            (name, self.get_step_item_dict(name)[step])
            for name in self.get_item_names()
            if step in self.get_step_item_dict(name)
        )

    def __len__(self):
        return len(self._steps)

    def get_items_count(self, item_name: str) -> int:
        self._raise_item_name_error(item_name)
        return len(self.get_step_item_dict(item_name))

    def _raise_item_name_error(self, item_name: str):
        if item_name not in self._values:
            raise ValueError(f"Item {item_name} not found.")

    def get_last_step(self) -> int:
        return self._steps[-1]

    def get_first_step(self) -> int:
        return self._steps[0]

    @staticmethod
    def to_serializable_params(model_parameters):
        return dict((k, str(v)) for k, v in model_parameters.items())

    def serialize_to(self, folder_path: str):
        """Creates `.crm` metadata file and `.drm` (or '.zrm' for compressed data) data file in specified path."""
        # experiment_id = self.model_name + '_' + str(list(self.model_parameters.values())) + '_' + get_stamp()
        file_name = to_safe_name(self.model_name + get_stamp())

        if self.use_zip:
            file_extension = '.zrm'
        else:
            file_extension = '.drm'

        drm_file_name = file_name + file_extension
        crm_file_name = file_name + '.ycrm'
        drm_file_path = path.join(folder_path, drm_file_name)
        crm_file_path = path.join(folder_path, crm_file_name)

        drm_file_path = to_safe_path(drm_file_path)
        crm_file_path = to_safe_path(crm_file_path)

        params_to_serialize = self.to_serializable_params(self.model_parameters)

        if self.use_zip:
            save_zip(self, drm_file_path, protocol=2)
        else:
            with open(drm_file_path, 'wb') as drm_file:
                pickle.dump(self, drm_file, protocol=2)

        with open(crm_file_path, 'w') as crm_file:
            yaml = YAML()
            yaml.dump({'model_name': self.model_name,
                       'model_parameters': params_to_serialize,
                       'data_file': drm_file_name},
                      crm_file)

    @staticmethod
    def _creation_date(path_to_file):
        path_to_file = to_safe_path(path_to_file)
        if platform.system() == 'Windows':
            return os.path.getctime(path_to_file)
        else:
            stat = os.stat(path_to_file)
            try:
                # noinspection PyUnresolvedReferences
                return stat.st_birthtime
            except AttributeError:
                return stat.st_mtime

    @staticmethod
    def browse_content_files(folder_path) -> Dict[Any, Any]:
        folder_path = to_safe_path(folder_path)
        data_files = {}
        file_paths = list(glob.iglob(path.join(folder_path, '*.crm'))) + \
                     list(glob.iglob(path.join(folder_path, '*.ycrm')))
        file_paths.sort(key=lambda x: SingleRunMeasurements._creation_date(x))
        for file_path in file_paths:
            if file_path.endswith('.ycrm'):
                with open(file_path, 'r') as file:
                    content_dict = yaml.load(file, Loader=yaml.Loader)
            else:
                with open(file_path, 'rb') as file:
                    content_dict = pickle.load(file)
                    content_dict['model_parameters'] = \
                        SingleRunMeasurements.to_serializable_params(content_dict['model_parameters'])
            ids = (content_dict['model_name'], str(content_dict['model_parameters']))
            data_files[ids] = {'model_parameters': content_dict['model_parameters'],
                               'data_file': path.join(path.dirname(file_path), content_dict['data_file'])}
        return data_files

    @staticmethod
    def load_from_data_file(drm_file_path: str):
        if drm_file_path.endswith('.zrm'):
            return load_zip(drm_file_path)
        else:
            drm_file_path = to_safe_path(drm_file_path)
            with open(drm_file_path, 'rb') as drm_file:
                return pickle.load(drm_file)


class TrainTestMeasurementPartitioning:
    def __init__(self, run_measurement: SingleRunMeasurements):
        self._run_measurement = run_measurement

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
        values = self._run_measurement.get_step_item_dict(item_name)
        if type(phase_source) == str:
            ids = self._run_measurement.get_step_item_tuples(phase_source)
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

    def partition_to_list_of_training_phases(self, item_name: str, remove_steps: bool = True) \
            -> Union[List[List[Tuple[int, Any]]], List[List[Any]]]:
        """
        Partition a measurement given by name to training phases
        Args:
            item_name: name of the measured variable
            remove_steps: if true, the method will return List[List[Any]]: for each phase list of measured values
        Returns: measured values partitioned into training phases
        """
        return self.partition_to_list_of_phases(item_name, phase_source='training_phase_id', remove_steps=remove_steps)

    def partition_to_list_of_testing_phases(self, item_name: str, remove_steps: bool = True) \
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
            return self._remove_steps(phases)
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
