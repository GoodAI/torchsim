import logging
import random
from typing import List, Dict

import torch

from torchsim.core.nodes import RandomNumberNode
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.node_groups.multilayer_model_group import MultilayerModelGroup

logger = logging.getLogger(__name__)


class DummyModelGroup(MultilayerModelGroup):
    """Just a dummy implementation of the interface which can be used for testing of the template
    """

    _is_learning: bool
    _num_layers: int

    _rand_node: RandomNumberNode

    _sim_step: int
    _testing_phase_id: int
    _testing_outputs_backup: Dict[int, List[torch.Tensor]]

    # for each layer, remember how many times this method has been called
    _num_abd_calls: Dict[int, int]
    _num_ad_calls: Dict[int, int]
    _num_boosted_clusters_calls: Dict[int, int]

    def __init__(self):
        super().__init__("Dummy Group")

        self._num_layers = 2
        self._layer_sizes = [11, 12]
        self._sp_size = 14

        self._rand_node = RandomNumberNode(upper_bound=20)
        self.add_node(self._rand_node)

        self._sim_step = 0
        self._testing_phase_id = -1

        # for each layer there is a backup list of output values which is stored during the first test phase
        self._testing_outputs_backup = {}

        self._num_abd_calls = {}
        self._num_ad_calls = {}
        self._num_boosted_clusters_calls = {}

        for layer_id in range(len(self._layer_sizes)):
            self._testing_outputs_backup[layer_id] = []
            self._num_abd_calls[layer_id] = 0
            self._num_ad_calls[layer_id] = 0
            self._num_boosted_clusters_calls[layer_id] = 0

    def _step(self):
        super()._step()
        self._sim_step += 1

    def get_average_log_delta_for(self, layer_id: int) -> float:
        self._num_ad_calls[layer_id] += 1
        return 0.1 + self._num_ad_calls[layer_id] + layer_id

    def get_average_boosting_duration_for(self, layer_id: int) -> float:
        self._num_abd_calls[layer_id] += 1
        return 0.2 + self._num_abd_calls[layer_id] + layer_id

    def get_num_boosted_clusters_ratio(self, layer_id: int) -> float:
        self._num_boosted_clusters_calls[layer_id] += 1
        return 1/(0.3 + self._num_boosted_clusters_calls[layer_id] + layer_id)

    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        """During testing, the labels are not available to this group, so output just random one-hot vectors"""
        return self._rand_node.outputs.one_hot_output.tensor.clone()

    def get_flock_size_of(self, layer_id: int) -> int:
        assert 0 <= layer_id < len(self._layer_sizes)
        if layer_id == 0:
            return 1
        return 2

    def get_sp_size_for(self, layer_id: int) -> int:
        assert 0 <= layer_id < len(self._layer_sizes)
        return self._layer_sizes[layer_id]

    def get_output_id_for(self, layer_id: int) -> int:
        assert layer_id == 1
        return random.randint(0, 19)

    def _should_load_backup(self, layer_id: int) -> bool:
        return self._testing_phase_id == (layer_id + 1)

    def clone_sp_output_tensor_for(self, layer_id: int) -> torch.Tensor:
        """Produces random outputs,

        only during the second testing phase, the results are identical to the first phase (cluster agreement=1)"""
        assert 0 <= layer_id < len(self._layer_sizes)

        # the template finds argmax, so it does not matter that the result is not layer_sizes{layer] times one-hot
        output = torch.rand(self._layer_sizes[layer_id], self._sp_size, device='cuda')

        if self._testing_phase_id == 0:
            # backup the outputs from the first phase
            self._testing_outputs_backup[layer_id].append(output)
        if self._should_load_backup(layer_id):
            # first layer loads backup during the second testing phase, second during third..
            output = self._testing_outputs_backup[layer_id].pop(0)

        return output

    def model_switch_to_training(self):
        self._is_learning = True

    def model_switch_to_testing(self):
        self._is_learning = False
        self._testing_phase_id += 1

    def is_learning(self):
        return self._is_learning

    def num_layers(self):
        return self._num_layers

