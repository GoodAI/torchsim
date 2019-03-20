import pytest

from torchsim.core import FLOAT_NEG_INF
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval.node_accessors.se_io_accessor import SeIoAccessor
from torchsim.core.graph import Topology
from torchsim.core.nodes.dataset_se_objects_node import DatasetSeObjectsParams, DatasetConfig, DatasetSeObjectsUnit
from torchsim.research.se_tasks.topologies.se_io.se_io_base import SeIoBase
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset import SeIoTask0Dataset


class EmptyTopology(Topology):
    _se_io: SeIoBase

    def __init__(self, params: DatasetSeObjectsParams = DatasetSeObjectsParams(), device: str = 'cuda'):
        super().__init__(device)

        self._se_io = self._get_installer(params)
        self._se_io.install_nodes(self)

    @staticmethod
    def _get_installer(params: DatasetSeObjectsParams):
        return SeIoTask0Dataset(params)


@pytest.mark.slow
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('start_training', [True, False])
def test_correct_dimensions(device, start_training):
    """Install the SeDataset with the installer and test the installer accessor gives correct data types."""

    params = DatasetSeObjectsParams()
    params.dataset_size = SeDatasetSize.SIZE_24
    params.save_gpu_memory = True
    if start_training:
        params.dataset_config = DatasetConfig.TRAIN_TEST
    else:
        params.dataset_config = DatasetConfig.TEST_ONLY

    t = EmptyTopology(params)

    if start_training:
        assert t._se_io.get_testing_phase() == 0.0
    else:
        assert t._se_io.get_testing_phase() == 1.0

    t.order_nodes()
    t._update_memory_blocks()
    t.step()
    t.step()

    landmark_id = SeIoAccessor.get_landmark_id_int(t._se_io.outputs)
    label_id = SeIoAccessor.get_label_id(t._se_io)

    assert type(landmark_id) is float
    assert landmark_id == FLOAT_NEG_INF

    assert type(label_id) is int
    assert label_id <= SeIoAccessor.get_num_labels(t._se_io)

    assert SeIoAccessor.get_num_labels(t._se_io) == DatasetSeObjectsUnit.NUM_LABELS

    if start_training:
        assert t._se_io.get_testing_phase() == 0.0
    else:
        assert t._se_io.get_testing_phase() == 1.0

    print('done')
