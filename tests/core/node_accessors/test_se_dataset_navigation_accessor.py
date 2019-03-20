import pytest
import torch

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval.node_accessors.se_io_accessor import SeIoAccessor
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.dataset_se_navigation_node import SamplingMethod, DatasetSENavigationParams, \
    DatasetSeNavigationNode


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('sampling_method', [e for e in SamplingMethod])
def test_dataset_accessor_return_type(device, sampling_method):
    """Should produce identical sequences, only if seed is None and random order is required, they should be unique."""
    params = DatasetSENavigationParams(SeDatasetSize.SIZE_24)

    # parametrized:
    params.sampling_method = sampling_method

    node = DatasetSeNavigationNode(params)
    node.allocate_memory_blocks(AllocatingCreator(device=device))

    node._step()

    # note: the actual node accessor has been deleted in favor of SeIoAccessor, the functionality remains worth testing
    landmark_id = SeIoAccessor.get_landmark_id_int(node.outputs)

    assert type(landmark_id) is int


