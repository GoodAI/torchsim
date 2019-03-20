import inspect
import logging
import os
from abc import abstractmethod, ABC
from copy import deepcopy
from os import path
from typing import List, Tuple, Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from eval_utils import progress_bar, garbage_collect
from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.measurement_manager import MeasurementManager
from torchsim.core.eval.run_measurement import RunMeasurement
from torchsim.core.eval.series_plotter import get_experiment_results_folder
from torchsim.core.eval.series_plotter import get_stamp, to_safe_name
from torchsim.core.eval.testable_measurement_manager import TestableMeasurementManager
from torchsim.core.eval.topology_adapter_base import TopologyAdapterBase, TestableTopologyAdapterBase
from torchsim.core.graph import Topology
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.utils.seed_utils import set_global_seeds
from torchsim.utils.template_utils.train_test_topology_saver import PersistableSaver

logger = logging.getLogger(__name__)


class ExperimentTemplateBase:
    """A thing which runs _n_runs of the same experiment (same or different seed) and collects the measurements.

    _measurements is a list of Measurement instances, where each has
        -an adapter which collects some data on the TopologySubject (by calling some NodeAdapter)
        -period of measurement
        -name

    after all the measurements are done, the statistics are computed from this, which can be plotted in the report.
    """

    _topology: Topology
    _max_steps: int
    _seed = 0

    rand: np.random

    _measurements: List[RunMeasurement]
    _disable_plt_show: bool

    _header_table_width = 60

    def __init__(self,
                 topology_adapter: TopologyAdapterBase,
                 topology_class,
                 models_params: List[Union[Tuple[Any], Dict[str, Any]]],
                 max_steps: int,
                 save_cache=False,
                 load_cache=False,
                 computation_only=False,
                 seed=None,
                 disable_plt_show=False,
                 experiment_folder=None,
                 clear_cache=True,
                 experiment_name=None):
        """Initialize.

        Args:
            disable_plt_show: if true, disables the plt.show called after each run (blocking the thread otherwise)
        """
        self._load_cache = load_cache
        self._save_cache = save_cache
        self._computation_only = computation_only
        self._init_seed(seed)
        self._max_steps = max_steps
        self._topology_adapter = topology_adapter
        self._topology_parameters_list = models_params
        self._topology_parameters = None
        self._topology_class = topology_class
        self._topology = None
        self._disable_plt_show = disable_plt_show
        self._experiment_name = experiment_name or self._experiment_template_name()

        self._experiment_folder, self._docs_folder = self._create_cache_folders(clear_cache,
                                                                                alternative_folder=experiment_folder)

    def _init_seed(self, seed: int):
        """Determines whether these measurements will be deterministic (across different runs)."""
        self.rand = np.random.RandomState()
        self.rand.seed(seed=seed)

        set_global_seeds(seed)

    def run(self):
        garbage_collect()  # empty the memory from previous experiments so that they do not influence this one
        print(f'---------------- collecting measurements')
        self._collect_measurements()
        if self._computation_only:
            return
        print(f'---------------- computing statistics')
        self._compute_experiment_statistics()
        print(f'---------------- plotting results')
        self._publish_results()
        print(f'---------------- done')
        if not self._disable_plt_show:
            plt.show()

    def _collect_measurements(self):
        """Should run _n_runs measurements, each with different initialization (or inputs)."""
        for topology_idx, topology_parameters in enumerate(self._topology_parameters_list):
            tqdm.write(f'--------------------------------------------------')
            tqdm.write(f'training topology no.{topology_idx}')
            tqdm.write(self.parameters_to_string([self._unpack_params(topology_parameters)])[0])
            garbage_collect()  # empty the memory from previous experiments 
            self._switch_topology(topology_parameters)
            self._run_learning_session()
            if not self._computation_only:
                self._after_run_finished()
            self._get_measurement_manager().finish_run()

    def _do_before_topology_step(self):
        """Override if you want to do something right before each topology step."""
        pass

    def _do_after_topology_step(self):
        """Override if you want to do something right after each topology step."""
        pass

    def _run_learning_session(self):
        """Runs one learning session, logs the ongoing results."""
        topology_name = self._topology_class.__name__

        if self._load_cache:
            if self._computation_only:
                loaded = self._get_measurement_manager().is_run_cached(topology_name, self._topology_parameters)
            else:
                loaded = self._get_measurement_manager().try_load_run(topology_name, self._topology_parameters)
            if loaded:
                tqdm.write("Loaded from cache.")
                return

        self._get_measurement_manager().create_new_run(topology_name, self._topology_parameters)

        iterator = progress_bar(range(self._max_steps), "progress: ")
        for step in iterator:
            # make one step of the topology
            self._do_before_topology_step()
            self._topology.step()
            self._do_after_topology_step()
            # make measurement
            self._get_measurement_manager().step(step)
            if self._should_end_run():
                iterator.close()
                break

        if self._save_cache:
            self._get_measurement_manager().cache_last_run()
            tqdm.write("Saved to cache.")

    def _should_end_run(self):
        """Should the run end prematurely?"""
        return False

    def _switch_topology(self, topology_parameters):
        del self._topology
        garbage_collect()

        if isinstance(topology_parameters, type({})):
            self._topology = self._topology_class(**topology_parameters)
        else:
            self._topology = self._topology_class(*topology_parameters)
        self._topology.prepare()
        self._topology_parameters = topology_parameters
        self._topology_adapter.set_topology(self._topology)

    def _create_cache_folders(self, clear_old_cache: bool = True, alternative_folder=None):
        if alternative_folder is None:
            experiment_folder = path.join(get_experiment_results_folder(), self._experiment_template_name())
        else:
            experiment_folder = alternative_folder

        if clear_old_cache:
            try:
                for file_name in os.listdir(experiment_folder):
                    if file_name.endswith('.drm') or file_name.endswith('.crm'):
                        os.unlink(path.join(experiment_folder, file_name))
            except FileNotFoundError as e:
                logger.warning("cache folder not found: \n" + str(e))
                pass

        if self._save_cache and not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)

        date = get_stamp()
        docs_folder = path.join(experiment_folder, "docs", self._complete_name() + date)

        if self._save_cache and not os.path.exists(docs_folder):
            os.makedirs(docs_folder)

        return experiment_folder, docs_folder

    def _create_measurement_manager(self, experiment_folder, delete_after_each_run=True, zip_data=False):
        if not self._save_cache and not self._load_cache:
            return MeasurementManager(None, delete_after_each_run)

        return MeasurementManager(experiment_folder, delete_after_each_run, zip_data_files=zip_data)

    @abstractmethod
    def _get_measurement_manager(self) -> MeasurementManager:
        pass

    @abstractmethod
    def _after_run_finished(self):
        """This method is called after each run is finished if computation_only flag is off."""
        pass

    @abstractmethod
    def _compute_experiment_statistics(self):
        """Should be called after all the measurements are collected.

        This converts measurements into computed values that can be plotted in the report.
        """

    def _prepare_document(self) -> (Document, str, os.path):
        """Prepares the document for writing the results"""
        doc = Document()
        date = get_stamp()

        doc.add(self._get_heading(date))
        return doc, date

    def _write_document(self, doc: Document, date: str, docs_folder: os.path):
        """ Writes the document to the hdd after adding the graph (see the _publish_results())."""
        doc.write_file(path.join(docs_folder, to_safe_name(self._complete_name() + date + ".html")))
        print('done')

    def _publish_results(self):
        """Plots the results into topologies and saves them.

        This is a complete thing which creates document, writes there and saves it to disc.
        """
        doc, date = self._prepare_document()
        self._publish_results_to_doc(doc, date, self._docs_folder)
        self._write_document(doc, date, self._docs_folder)

    def _publish_results_to_doc(self, doc: Document, date: str, docs_folder: os.path):
        """An alternative to the _publish_results method, this is called from _publish_results now

        Draw and add your topologies to the document here.
        """
        pass

    @abstractmethod
    def _experiment_template_name(self):
        """Reasonably unique experiment name for storing the results to the common folder."""

    @classmethod
    def parameters_to_string(cls, parameters: List[Dict[str, Any]]) -> List[str]:

        return [", ".join(
            f"{param}: {cls._param_value_to_string(value)}" for param, value in parameter.items()
            if "training_phase_steps" not in param and "testing_phase_steps" not in param
        ) for parameter in parameters]

    @classmethod
    def _param_value_to_string(cls, value) -> str:
        """Lists of normal values are parsed OK, param value can be also list of classes, parse to readable string.
        """
        if type(value) in (list, tuple):
            list_string = [cls._param_value_to_string(x) for x in value]
            return '['+', '.join(list_string)+']'  # format the list nicely
        elif isinstance(value, type):
            return value.__name__
        return str(value)

    @staticmethod
    def _remove_params(params: List[Dict[str, Any]], params_to_remove: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = params.copy()

        for run_param in result:
            for key, _ in params_to_remove.items():
                if key in run_param:
                    del run_param[key]
                else:
                    raise ValueError(f'key {key} not found in the params, params should be complete now!')

        return result

    @staticmethod
    def _find_constant_parameters(params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify all the parameters that are not changing across different runs

        return the dictionary with key: constant_value
        """
        # remove anything incompatible with this list
        constant_params = params[0]
        filtered = constant_params.copy()

        for run_params in params:
            for key, value in constant_params.items():
                if key not in run_params:
                    raise ValueError(f'Param named {key} not found in one of runs.'
                                     f' But all runs should have complete params at this point!')
                if run_params[key] != value:
                    del filtered[key]

            constant_params = filtered.copy()

        return constant_params

    @staticmethod
    def _get_default_topology_params(topology_class) -> Dict[str, Any]:
        """
        Get default params of the topology - extracted from the constructor
        Args:
            topology_class: topology class to be measured

        Returns: dictionary {param_name: param_default_value}
        """
        my_params = dict(inspect.signature(topology_class.__init__).parameters)

        params_dict = dict((key, value.default)
                           for key, value in zip(my_params.keys(), list(my_params.values()))
                           if key is not 'self')

        return params_dict

    @staticmethod
    def add_common_params(params: List[Dict], common_params: Dict) -> List[Dict]:
        """Add the dictionary of common parameters to the list of parameters for each run

        params with the key already defined in the dict are not overwritten by the common_params.
        """

        result = deepcopy(params)

        # iterate through each run
        for param in result:
            # add each key->value pair from the common_params to this one
            for key, value in common_params.items():
                if key not in param:
                    param[key] = value

        return result

    @staticmethod
    def _extract_params_for_heading(params: List[Dict[str, Any]], topology_class) -> Dict[str, Any]:
        """ Extract all parameters should be in the heading.

        The heading should contain the following parameters:
            -parameters identical for all runs (from the experiment params and from the constructor)
            -default parameters that are not overwritten in all experiment runs
        Args:
            params: parameters defined in the experiment
            topology_class: subject of the experiment
        Returns: dictionary: {param_name: common_param_value]
        """

        topology_params = ExperimentTemplateBase._get_default_topology_params(topology_class)
        topology_params = ExperimentTemplateBase._unpack_params(topology_params)

        all_run_params = [ExperimentTemplateBase._unpack_params(run_params) for run_params in params]

        all_params = ExperimentTemplateBase.add_common_params(all_run_params, topology_params)

        constant_params = ExperimentTemplateBase._find_constant_parameters(all_params)
        return constant_params

    @classmethod
    def param_to_html(cls, name: str, value: Any) -> str:

        # val = ExperimentTemplateBase._param_value_to_string(value)

        row = "<tr>" + \
              f"<td>{name}</td>" + \
              f"<td>{cls._param_value_to_string(value)}</td>" + \
              "</tr>"

        return row

    def template_configuration_to_html(self) -> str:
        """Convert the configuration of this template to the string for the html heading.

        Override this in order to add your params to the header.
        """
        result = f"\n<br><b>Template configuration</b>:<br> "
        result += self._get_table_header()

        # values in columns
        result += self.param_to_html("max_steps", self._max_steps)
        result += "</table>"

        return result

    def _get_table_header(self):
        header = f"<table style=\"width:{self._header_table_width}%\">" + \
                 "<tr>" + \
                 "<th style=\"text-align: left\">Param name</th>" + \
                 "<th style=\"text-align:left\">Param value</th>" + \
                 "</tr>"
        return header

    @staticmethod
    def _unpack_params(params: Dict[str, Any]) -> Dict[str, Any]:
        unpacked_params = {}
        for key, value in params.items():
            if isinstance(value, MultipleLayersParams):
                # get dictionary {prefix+short_name: value} and merge with the rest
                unpacked_params = {**unpacked_params, **value.get_params_as_short_names(prefix=key[0:4]+"_")}
            else:
                unpacked_params[key] = value

        return unpacked_params

    @classmethod
    def extract_params_for_legend(cls, params: List[Dict[str, Any]]) -> List[str]:
        """ Extract parameters that should be shown in the legend (just those that are not identical for all runs).
        Args:
            params: params that are passed to the ExperimentTemplate
        Returns: List of strings, each item for one run
        """

        # unpack params classes as individual params
        unpacked_params = [cls._unpack_params(param_set) for param_set in params]

        # find params identical for all runs and remove them from the lists
        constant_params = ExperimentTemplateBase._find_constant_parameters(unpacked_params)
        changing_params = ExperimentTemplateBase._remove_params(unpacked_params, constant_params)

        return ExperimentTemplateBase.parameters_to_string(
            changing_params
        )

    def _complete_name(self):
        return f"{self._topology_class.__name__}_" + self._experiment_name + "_"

    def _get_heading(self, date: str):
        """Get heading of the html file with the experiment description"""

        info = f"<p><b>Template</b>: {type(self).__name__}<br>" + \
               f"\n<b>topology:</b> {self._topology_class.__name__}<br>" + \
               f"\n<b>Experiment_name</b>: {self._experiment_name}<br>" + \
               f"\n<b>Date:</b> {date[1:]}</p>" + \
               f"\n<b>List of common parameters</b>:<br> "

        # get the params that should be shown in the header
        all_common_params = ExperimentTemplateBase._extract_params_for_heading(self._topology_parameters_list,
                                                                               self._topology_class)
        # create table with the params
        info += self._get_table_header()
        for key, value in all_common_params.items():
            info += self.param_to_html(key, value)
        info += "</table>"

        # add the description of the template configuration
        info += self.template_configuration_to_html()

        return info


class TestableExperimentTemplateBase(ExperimentTemplateBase, ABC):
    _topology_adapter: TestableTopologyAdapterBase

    _training_step: int  # ID of the currently performed step (the one that will be done after _do_before_topology_step)
    _testing_step: int
    _training_phase_id: int  # ID of the currently performed phase
    _testing_phase_id: int
    _overall_training_steps: int

    _training_phase: bool  # whether we are in the training phase now

    _topology_saver: PersistableSaver

    def __init__(self,
                 topology_adapter: TestableTopologyAdapterBase,
                 topology_class,
                 models_params: List[Union[Tuple[Any], Dict[str, Any]]],
                 overall_training_steps: int,
                 num_testing_steps: int,
                 num_testing_phases: int,
                 save_cache=False,
                 load_cache=False,
                 computation_only=False,
                 seed=None,
                 disable_plt_show=False,
                 experiment_folder=None,
                 experiment_name=None,
                 clear_cache=True):
        super().__init__(topology_adapter,
                         topology_class,
                         models_params,
                         overall_training_steps + num_testing_steps * num_testing_phases,
                         save_cache,
                         load_cache,
                         computation_only,
                         seed,
                         disable_plt_show,
                         experiment_name=experiment_name,
                         experiment_folder=experiment_folder,
                         clear_cache=clear_cache)

        self._num_testing_phases = num_testing_phases
        self._num_testing_steps = num_testing_steps

        self._overall_training_steps = overall_training_steps

        self._training_steps_between_testing = overall_training_steps // self._num_testing_phases

        self._topology_saver = PersistableSaver(type(topology_adapter).__name__)

    def _do_before_topology_step(self):
        ExperimentTemplateBase._do_before_topology_step(self)

        if self._training_phase:
            # decide whether to do the training for: self._training_step + 1
            if (self._training_step + 1) % self._training_steps_between_testing == 0 and self._training_step != -1:
                self._topology_saver.save_data_of(self._topology)

                self._topology_adapter.switch_to_testing()
                self._training_phase = False
                self._testing_phase_id += 1  # ID of the current phase being measured
                self._handle_switching_to_testing_phase()
        else:
            if (self._testing_step + 1) % self._num_testing_steps == 0 and self._testing_step != -1:
                self._topology_saver.load_data_into(self._topology)

                self._topology_adapter.switch_to_training()
                self._training_phase = True
                self._training_phase_id += 1
                self._handle_switching_to_training_phase()

        if self._training_phase:
            self._training_step += 1
        else:
            self._testing_step += 1

    def _handle_switching_to_testing_phase(self):
        """Add a custom functionality if necessary.

        E.g. in case the testing phase should have multiple pars etc..
        """
        pass

    def _handle_switching_to_training_phase(self):
        """Add a custom functionality if necessary."""
        pass

    def _create_measurement_manager(self, experiment_folder, delete_after_each_run=True, zip_data=False):
        if not self._save_cache and not self._load_cache:
            cache_folder = None
            should_zip = False
        else:
            cache_folder = experiment_folder
            should_zip = zip_data

        return TestableMeasurementManager(self._topology_adapter.is_in_training_phase,
                                          self._testing_phase_id_f,
                                          self._training_phase_id_f,
                                          self._testing_step_f,
                                          self._training_step_f,
                                          cache_folder=cache_folder,
                                          delete_after_each_run=delete_after_each_run,
                                          zip_data_files=should_zip)

    def _testing_phase_id_f(self):
        if self._training_phase:
            return -1
        return self._testing_phase_id

    def _training_phase_id_f(self):
        if not self._training_phase:
            return -1
        return self._training_phase_id

    def _testing_step_f(self):
        if self._training_phase:
            return -1
        return self._testing_step

    def _training_step_f(self):
        if not self._training_phase:
            return -1
        return self._training_step

    def _switch_topology(self, topology_parameters):
        # this prepares the topology
        ExperimentTemplateBase._switch_topology(self, topology_parameters)
        self._testing_phase_id = -1
        self._testing_step = -1

        self._topology_adapter.switch_to_training()
        self._training_phase = True
        self._training_phase_id = 0
        self._training_step = -1

    def template_configuration_to_html(self) -> str:
        """Convert the configuration of this template to the string for the html heading."""

        result = f"\n<br><b>Template configuration</b>:"
        result += self._get_table_header()

        # values in columns
        result += self.param_to_html("max_steps", self._max_steps)
        result += self.param_to_html("overall_training_steps", self._overall_training_steps)
        result += self.param_to_html("num_testing_steps", self._num_testing_steps)
        result += self.param_to_html("num_testing_phases", self._num_testing_phases)
        result += "</table>"

        return result


class TaskExperimentStatistics:
    """Information about the status of a task during an experiment."""

    def __init__(self, task_id: int):
        self.task_id = task_id
        self.task_solved = None
        self.instances_seen_training = 0
        self.instances_solved_training = 0
        self.instances_seen_testing = 0
        self.instances_solved_testing = 0

    def add_instance(self, solved: bool, testing_phase: bool):
        if testing_phase:
            self.instances_seen_testing += 1
            if solved:
                self.instances_solved_testing += 1
        else:
            self.instances_seen_training += 1
            if solved:
                self.instances_solved_training += 1

    def set_task_solved(self, solved: bool):
        self.task_solved = solved
