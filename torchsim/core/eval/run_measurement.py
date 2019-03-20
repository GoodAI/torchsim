from collections import OrderedDict

import glob
import gzip
import os
import pickle
import platform
from os import path
from ruamel import yaml
from ruamel.yaml import YAML
from typing import List, Any, Dict, Tuple

from torchsim.core.eval.series_plotter import get_stamp, to_safe_path


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


class RunMeasurement:
    """Class for easy collecting and retrieval of measured values.

    Designed to work with one run - when adding, `step` value must always increase.
    """
    model_name: str
    model_parameters: Dict[str, Any]

    def __init__(self, model_name: str = '', model_parameters: Dict[str, Any] = None, zip_files: bool=False):
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
        self._values['current_step'].append((step, step))

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
        file_name = self.model_name + get_stamp()

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
        file_paths.sort(key=lambda x: RunMeasurement._creation_date(x))
        for file_path in file_paths:
            if file_path.endswith('.ycrm'):
                with open(file_path, 'r') as file:
                    content_dict = yaml.load(file, Loader=yaml.Loader)
            else:
                with open(file_path, 'rb') as file:
                    content_dict = pickle.load(file)
                    content_dict['model_parameters'] = \
                        RunMeasurement.to_serializable_params(content_dict['model_parameters'])
            ids = (content_dict['model_name'], str(content_dict['model_parameters']))
            data_files[ids] = path.join(path.dirname(file_path), content_dict['data_file'])
        return data_files

    @staticmethod
    def load_from_data_file(drm_file_path: str):
        if drm_file_path.endswith('.zrm'):
            return load_zip(drm_file_path)
        else:
            drm_file_path = to_safe_path(drm_file_path)
            with open(drm_file_path, 'rb') as drm_file:
                return pickle.load(drm_file)
