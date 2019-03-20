import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict, Any

from torchsim.core.eval.experiment_template_base import TestableExperimentTemplateBase
from torchsim.core.eval.measurement_manager import MeasurementManagerBase
from torchsim.core.eval.topology_adapter_base import TestableTopologyAdapterBase


class Task0TrainTestTemplateAdapterBase(TestableTopologyAdapterBase, ABC):
    """Provide access to all layers of the topology, provide functionality to switch train/test etc."""

    @abstractmethod
    def get_label_id(self) -> int:
        """Should return ID indicating current class label, from range <0, NO_CLASSES)."""
        pass

    @abstractmethod
    def clone_ground_truth_label_tensor(self) -> torch.Tensor:
        """Returns a tensor containing the current label, but not hidden during testing."""
        pass

    @abstractmethod
    def clone_constant_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        """Returns a tensor with length of num_labels, representing prediction of class label from random baseline."""
        pass

    @abstractmethod
    def clone_random_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        """The get_baseline_output_tensor_for_labels is from constant baseline, this one is random baseline."""
        pass


class Task0TrainTestTemplateBase(TestableExperimentTemplateBase):
    """Usefulness of representation of various topologies on the Task0, measured using the train/test split."""

    _measurement_period: int
    _sliding_window_size: int
    _sliding_window_stride: int

    _layer_measurement_managers: List

    def __init__(self,
                 topology_adapter: Task0TrainTestTemplateAdapterBase,
                 topology_class,
                 topology_parameters: List[Union[Tuple[Any], Dict[str, Any]]],
                 overall_training_steps: int,
                 num_testing_steps: int,
                 num_testing_phases: int,
                 measurement_period: int = 5,
                 sliding_window_size: int = 9,
                 sliding_window_stride: int = 5,
                 sp_evaluation_period: int = 10,
                 seed=None,
                 experiment_name: str = 'empty_name',
                 save_cache=True,
                 load_cache=True,
                 clear_cache=True,
                 computation_only=False,
                 experiment_folder=None,
                 disable_plt_show=True):
        """Initialize.

        Support to:
         -collect data in train/test phases,
         -switching between train/test,
         -train/test scheduling configuration.
        """
        super().__init__(topology_adapter=topology_adapter,
                         topology_class=topology_class,
                         models_params=topology_parameters,
                         overall_training_steps=overall_training_steps,
                         num_testing_steps=num_testing_steps,
                         num_testing_phases=num_testing_phases,
                         save_cache=save_cache,
                         load_cache=load_cache,
                         computation_only=computation_only,
                         seed=seed,
                         disable_plt_show=disable_plt_show,
                         experiment_folder=experiment_folder,
                         experiment_name=experiment_name,
                         clear_cache=clear_cache)
        # measurement frequencies
        self._sliding_window_stride = sliding_window_stride
        self._sp_evaluation_period = sp_evaluation_period
        self._measurement_period = measurement_period
        self._sliding_window_size = sliding_window_size

        # more general ones
        self._measurement_manager = self._create_measurement_manager(self._experiment_folder,
                                                                     delete_after_each_run=False)

        assert self._training_steps_between_testing % self._sp_evaluation_period == 0, \
            f"training_steps_between_testing ({self._training_steps_between_testing}) should be divisible " + \
            f"by sp_evaluation_period ({self._sp_evaluation_period})"

        assert self._training_steps_between_testing % self._measurement_period == 0, \
            f"training_steps_between_testing ({self._training_steps_between_testing}) should be divisible " + \
            f"by measurement_period ({self._measurement_period})"

        assert self._num_testing_steps % self._sp_evaluation_period == 0, \
            f"num_testing_steps ({self._num_testing_steps}) should be divisible " + \
            f"by sp_evaluation_period ({self._sp_evaluation_period})"

        assert self._num_testing_steps % self._measurement_period == 0, \
            f"num_testing_steps ({self._num_testing_steps}) should be divisible " + \
            f"by _measurement_period ({self._measurement_period})"

        # add the layer measurement manager if needed here
        # add the rest of the measurements here

    def _get_measurement_manager(self) -> MeasurementManagerBase:
        return self._measurement_manager

    def _experiment_template_name(self):
        return self._experiment_name

    @abstractmethod
    def _after_run_finished(self):
        pass

    @abstractmethod
    def _compute_experiment_statistics(self):
        pass

