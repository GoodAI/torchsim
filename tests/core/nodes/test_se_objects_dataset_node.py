from copy import copy
from typing import List
import torch

from torchsim.core.datasets.dataset_se_base import DatasetSeBase, SeDatasetSize
from torchsim.core.graph import Topology
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.dataset_se_objects_node import DatasetSeObjectsParams, DatasetSeObjectsNode, DatasetConfig, \
    DatasetSeObjectsUnit
from torchsim.core.utils.tensor_utils import same
from torchsim.research.experiment_templates.task0_train_test_template_base import Task0TrainTestTemplateAdapterBase
from torchsim.topologies.SeDatasetObjectsTopology import SeDatasetObjectsTopology
from torchsim.utils.template_utils.train_test_topology_saver import PersistableSaver


def collect_data(node: DatasetSeObjectsNode, n_samples: int):
    """Collects n_samples from multiple steps of the DatasetSeObjectsNode and returns the outputs in lists"""
    labels = []
    labels_gt = []
    images = []

    for step in range(0, n_samples):
        node.step()

        image = node.outputs.image_output.tensor  # TODO unresolved reference indication?
        label = node.outputs.task_to_agent_label.tensor
        label_gt = node.outputs.task_to_agent_label_ground_truth.tensor

        # TODO collect the metadata outputs ???

        images.append(image.clone())
        labels.append(label.clone())
        labels_gt.append(label_gt.clone())

    return images, labels, labels_gt


def compare_sequences(labels_a: List[torch.Tensor], labels_b: List[torch.Tensor], epsilon: float = None) -> bool:

    if len(labels_a) != len(labels_b):
        return False

    for label_a, label_b in zip(labels_a, labels_b):
        if not same(label_a, label_b, epsilon):
            return False

    return True


def test_class_filter_and_data_sizes():
    """Collect n_steps outputs and verify their shapes and check that the class filter works."""

    device = 'cuda'

    n_steps = 20

    params = DatasetSeObjectsParams()
    params.dataset_size = SeDatasetSize.SIZE_24
    params.save_gpu_memory = True
    params.class_filter = [1, 2, 19, 5]
    params.dataset_config = DatasetConfig.TRAIN_ONLY
    params.random_order = True

    node = DatasetSeObjectsNode(params, seed=None)

    node.allocate_memory_blocks(AllocatingCreator(device))
    images, labels, labels_gt,  = collect_data(node, n_steps)

    # labels and labels_gt should be equal in the training phase
    assert compare_sequences(labels, labels_gt)

    assert images[0].device.type == device
    assert labels[0].device.type == device
    assert labels_gt[0].device.type == device

    assert images[0].shape[0] == SeDatasetSize.SIZE_24.value
    assert images[0].shape[1] == SeDatasetSize.SIZE_24.value
    assert images[0].shape[2] == DatasetSeBase.N_CHANNELS

    assert labels[0].shape[0] == DatasetSeObjectsUnit.NUM_LABELS

    # go through all of the class labels and check if each is contained in the filter
    for label in labels:
        _, max_id = label.max(0)
        max_id = max_id.item()

        assert max_id in params.class_filter


def train_pos(node: DatasetSeObjectsNode):
    return node.memory_blocks.training_pos.tensor.item()


def get_common_params() -> DatasetSeObjectsParams:
    params = DatasetSeObjectsParams()
    params.dataset_size = SeDatasetSize.SIZE_24
    params.save_gpu_memory = True
    params.class_filter = None
    params.dataset_config = DatasetConfig.TRAIN_ONLY
    params.random_order = False
    return params


def test_train_test_position_persistence():
    """If we switch from train to test, do some testing and then back to train, we should continue where stopped"""

    device = 'cuda'

    params = get_common_params()
    params.switch_train_resets_train_pos = False

    node = DatasetSeObjectsNode(params, seed=None)
    node.allocate_memory_blocks(AllocatingCreator(device))

    node.switch_training(training_on=True)
    assert node._unit._pos == -1
    first_steps = 10

    _, _, _,  = collect_data(node, first_steps)

    # we are at the expected position
    last_train_pos = -1 + first_steps
    assert node._unit._pos == last_train_pos
    assert train_pos(node) == last_train_pos

    # switch to training (redundant) and check position not changed
    node.switch_training(training_on=True)
    assert node._unit._pos == last_train_pos
    assert train_pos(node) == last_train_pos

    second_steps = 3
    # check the training position continues increasing
    _, _, _ = collect_data(node, second_steps)
    last_train_pos += second_steps
    assert node._unit._pos == last_train_pos
    assert train_pos(node) == last_train_pos

    # switch to testing and collect some data
    # _pos should reset (start testing), but the training_pos in the tensor should stay the same
    node.switch_training(training_on=False)
    assert node._unit._pos == -1
    assert train_pos(node) == last_train_pos
    testing_steps = 7

    test_images, test_labels, _ = collect_data(node, testing_steps)
    assert node._unit._pos == -1 + testing_steps  # pos in the training set changes
    assert train_pos(node) == last_train_pos  # train pos in the tensor does not

    # switch back to training and check expected positions
    third_train_steps = 2
    node.switch_training(training_on=True)
    _, _, _ = collect_data(node, third_train_steps)
    last_train_pos += third_train_steps
    assert train_pos(node) == last_train_pos
    assert node._unit._pos == last_train_pos

    # one more testing
    node.switch_training(training_on=False)
    test_images_2, test_labels_2, _ = collect_data(node, testing_steps)
    assert same(test_images[0], test_images_2[0])
    assert same(test_labels[0], test_labels_2[0])


def test_train_test_position_reset():
    """If we switch from train to test, we can start training from the beginning every time"""

    device = 'cuda'

    params = get_common_params()
    params.switch_train_resets_train_pos = True

    node = DatasetSeObjectsNode(params, seed=None)
    node.allocate_memory_blocks(AllocatingCreator(device))

    node.switch_training(training_on=True)
    assert node._unit._pos == -1
    first_steps = 10

    train_images, train_labels, _,  = collect_data(node, first_steps)

    # we are at the expected position
    last_train_pos = -1 + first_steps
    assert node._unit._pos == last_train_pos
    assert train_pos(node) == last_train_pos

    # switch to training (redundant) and check position is reset
    node.switch_training(training_on=True)
    assert node._unit._pos == -1
    assert train_pos(node) == -1

    second_steps = 3
    # check the training position starts from the beginning
    _, _, _ = collect_data(node, second_steps)
    last_train_pos = second_steps - 1
    assert node._unit._pos == last_train_pos
    assert train_pos(node) == last_train_pos

    # switch to testing and collect some data
    # _pos should reset (start testing), but the training_pos in the tensor should stay the same
    node.switch_training(training_on=False)
    assert node._unit._pos == -1
    assert train_pos(node) == last_train_pos
    testing_steps = 7

    test_images, test_labels, _ = collect_data(node, testing_steps)
    assert node._unit._pos == -1 + testing_steps  # pos in the training set changes
    assert train_pos(node) == last_train_pos  # train pos in the tensor does not

    # switch back to training and check expected positions (training pos should be resetted)
    third_train_steps = 2
    node.switch_training(training_on=True)
    assert node._unit._pos == -1
    assert train_pos(node) == -1

    train_images_2, train_labels_2, _ = collect_data(node, third_train_steps)

    last_train_pos = third_train_steps - 1
    assert train_pos(node) == last_train_pos
    assert node._unit._pos == last_train_pos

    # check that the data read at the beginning of training are equal
    assert same(train_images[0], train_images_2[0])
    assert same(train_labels[0], train_labels_2[0])


def test_save_gpu():
    """Collect n-steps outputs and verify that the sequences produced with save_gpu_memory true/false are the same."""

    device = 'cuda'

    n_steps = 20

    false_params = DatasetSeObjectsParams()
    false_params.dataset_size = SeDatasetSize.SIZE_24
    false_params.save_gpu_memory = True
    false_params.dataset_config = DatasetConfig.TRAIN_ONLY
    false_params.random_order = False
    params = DatasetSeObjectsParams()
    params.dataset_size = SeDatasetSize.SIZE_24
    params.dataset_config = DatasetConfig.TRAIN_ONLY
    params.random_order = False

    node = DatasetSeObjectsNode(false_params, seed=None)
    params_1 = copy(params)
    params_1.save_gpu_memory = True
    node = DatasetSeObjectsNode(params_1, seed=None)
    node.allocate_memory_blocks(AllocatingCreator(device))
    gpu_true_images, gpu_true_labels, labels_gt, = collect_data(node, n_steps)

    params_2 = copy(params)
    params_2.save_gpu_memory = False
    node = DatasetSeObjectsNode(params_2, seed=None)
    node.allocate_memory_blocks(AllocatingCreator(device))
    gpu_false_images, gpu_false_labels, labels_gt, = collect_data(node, n_steps)

    # note that here the epsilon is required (save_gpu introduces some rounding problems here)
    assert compare_sequences(gpu_true_images, gpu_false_images, epsilon=0.0001)
    assert compare_sequences(gpu_true_labels, gpu_false_labels, epsilon=0.0001)


def test_train_cycles():
    """Set the training position to the last element and make step. Then test weather the cycle starts in the
    begining. """

    device = 'cuda'

    # init node
    params = DatasetSeObjectsParams()
    params.dataset_size = SeDatasetSize.SIZE_24
    params.dataset_config = DatasetConfig.TRAIN_ONLY
    params.random_order = False
    params.save_gpu_memory = False

    node = DatasetSeObjectsNode(params, seed=None)
    node.allocate_memory_blocks(AllocatingCreator(device))

    # make step
    node.step()

    # read outputs
    first_image = node.outputs.image_output.tensor
    first_label = node.outputs.task_to_agent_label.tensor

    # hack the position and make another step
    node._unit.training_pos[0] = len(node._unit._train_images)
    node.step()

    # read another outputs
    output_image = node.outputs.image_output.tensor
    output_label = node.outputs.task_to_agent_label.tensor

    # assert they are both the same
    assert same(first_image, output_image)
    assert same(first_label, output_label)


def test_test_cycles():
    """Set the testing position to the last element and make step. Then test weather the cycle starts in the
    begining. """

    device = 'cuda'

    # init node
    params = DatasetSeObjectsParams()
    params.dataset_size = SeDatasetSize.SIZE_24
    params.dataset_config = DatasetConfig.TEST_ONLY
    params.random_order = False
    params.save_gpu_memory = False

    node = DatasetSeObjectsNode(params, seed=None)
    node.allocate_memory_blocks(AllocatingCreator(device))

    # make step
    node.step()

    # read outputs
    first_image = node.outputs.image_output.tensor
    first_label = node.outputs.task_to_agent_label_ground_truth.tensor

    # hack the position and make another step
    node._unit._pos = len(node._unit._test_images)
    node.step()

    # read another outputs
    output_image = node.outputs.image_output.tensor
    output_label = node.outputs.task_to_agent_label_ground_truth.tensor

    # assert they are both the same
    assert same(first_image, output_image)
    assert same(first_label, output_label)


class SeDatasetObjectsTopologyTest(SeDatasetObjectsTopology):
    """A topology with dataset for Task0 which remembers last training position in the dataset"""

    def restart(self):
        pass

    def __init__(self):
        params = get_common_params()
        params.switch_train_resets_train_pos = False
        super().__init__(params)


class SeDatasetObjectsTopologyTestAdapter(Task0TrainTestTemplateAdapterBase):
    """An adapter for the topology above, which supports switching between train/test phases"""

    def get_label_id(self) -> int:
        return -1

    def clone_ground_truth_label_tensor(self) -> torch.Tensor:
        """Label - one-hot tensor"""
        return self._dataset.outputs.task_to_agent_label_ground_truth.tensor.clone()

    def clone_constant_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        return None

    def clone_random_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        return None

    _topology: SeDatasetObjectsTopologyTest
    _dataset: DatasetSeObjectsNode

    def get_current_step(self) -> int:
        return -1

    def get_title(self) -> str:
        return 'unit_test_adapter'

    def get_average_log_delta_for(self, layer_id: int) -> float:
        return -1.0

    def is_in_training_phase(self, **kwargs) -> bool:
        return self._dataset.is_training()

    def switch_to_training(self):
        self._dataset.switch_training(training_on=True)

    def switch_to_testing(self):
        self._dataset.switch_training(training_on=False)

    def get_topology(self) -> Topology:
        return self._topology

    def set_topology(self, topology) -> Topology:
        self._topology = topology
        self._dataset = topology._node_se_dataset


def make_n_steps(topology: Topology, num_steps: int):
    for step in range(num_steps):
        topology.step()


def test_topology_train_save_test_load_train():
    """Simulation of train/test cycle in the TestableExperimentTemplateBase.

    This simulates:
        setup -> switch_to_train -> train -> save -> switch_to_test -> test -> switch_to_train -> train

    The second training should continue where the first one ended.
    """

    # setup the template 'mock'
    topology = SeDatasetObjectsTopologyTest()
    node = topology._node_se_dataset

    adapter = SeDatasetObjectsTopologyTestAdapter()
    adapter.set_topology(topology)

    saver = PersistableSaver(type(adapter).__name__)

    # prepare the simulation
    topology.prepare()
    adapter.switch_to_training()

    # training -> save
    make_n_steps(topology, 5)
    end_training_pos = train_pos(node)  # remember where we ended
    saver.save_data_of(topology)
    assert end_training_pos == train_pos(node)  # save does not mess up with it

    # testing
    testing_positions_1, testing_ground_truth_tensors_1 = run_testing_phase(adapter, topology)

    # load -> switch_to_train -> train
    saver.load_data_into(topology)
    assert end_training_pos == train_pos(node)
    adapter.switch_to_training()
    assert end_training_pos == train_pos(node)
    make_n_steps(topology, 3)
    # the only thing which should change the position is making steps in training
    assert end_training_pos + 3 == train_pos(node)

    # test again:
    testing_positions_2, testing_ground_truth_tensors_2 = run_testing_phase(adapter, topology)

    assert testing_positions_1 == testing_positions_2
    assert all([(tensors[0].equal(tensors[1]))
                for tensors in zip(testing_ground_truth_tensors_1, testing_ground_truth_tensors_2)])

    # additional tests:
    saver.load_data_into(topology)
    assert end_training_pos == train_pos(node)  # loading the topology loads original training pos


def run_testing_phase(adapter: SeDatasetObjectsTopologyTestAdapter, topology: SeDatasetObjectsTopologyTest) \
        -> (List[int], List[torch.Tensor]):
    testing_positions = []
    testing_ground_truth_tensors = []
    adapter.switch_to_testing()
    for _ in range(7):
        topology.step()
        testing_positions.append(topology._node_se_dataset._unit._pos)
        testing_ground_truth_tensors.append(adapter.clone_ground_truth_label_tensor())
    return testing_positions, testing_ground_truth_tensors

