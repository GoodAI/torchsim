from typing import List, Optional

from torchsim.core.models.expert_params import SamplingMethod
from torchsim.core.nodes import SeObjectsTaskPhaseParams, PhasedSeObjectsTaskParams
from torchsim.core.nodes import DatasetSeObjectsParams, DatasetConfig
from torchsim.research.research_topics.rt_2_1_1_relearning.topologies.task0_basic_topology import SeT0BasicTopologyRT211
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset_phased import SeIoTask0DatasetPhased

mock_se_objects_task_phase_params = [SeObjectsTaskPhaseParams([1, 2, 3, 4], 1.0, False, 100)]


class SeT0BasicTopologyRT211Phased(SeT0BasicTopologyRT211):
    """A model which receives data from the 0th SE task and learns spatial patterns."""

    se_io: SeIoTask0DatasetPhased

    def __init__(self, num_ccs: int = 20, buffer_size: int = 1000,
                 sampling_method: SamplingMethod = SamplingMethod.BALANCED, run_init=True,
                 phase_params: Optional[List[SeObjectsTaskPhaseParams]] = None,
                 use_dataset=True):
        super().__init__(run_init=False)  # force use dataset

        if phase_params is None:
            phase_params = mock_se_objects_task_phase_params

        self._num_ccs = num_ccs
        self._buffer_size = buffer_size
        self._sampling_method = sampling_method
        self._phase_params = phase_params

        if run_init:  # a small hack to allow to postpone init until children have set their parameters
            self.create_se_io(curriculum=(0,), use_dataset=True, save_gpu_memory=True, class_filter=[],
                              location_filter=1.0)
            self.init()

    def create_se_io(self, curriculum: tuple, use_dataset: bool, save_gpu_memory: bool, class_filter: List[int],
                     location_filter: float):
        params_dataset = DatasetSeObjectsParams(dataset_config=DatasetConfig.TRAIN_ONLY,
                                                save_gpu_memory=save_gpu_memory,
                                                class_filter=class_filter,
                                                location_filter_ratio=location_filter)

        self.se_io = SeIoTask0DatasetPhased(PhasedSeObjectsTaskParams(self._phase_params, params_dataset))
