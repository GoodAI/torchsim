import pytest
import torch

from torchsim.core import get_float
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.invertible_node import InversePassOutputPacket
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.core.utils.tensor_utils import same


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_inverse_projection(device):

    float_dtype = get_float(device)
    params = ExpertParams()
    params.flock_size = 2
    params.n_cluster_centers = 4

    params.spatial.input_size = 6
    params.spatial.buffer_size = 7
    params.spatial.batch_size = 3
    input_size = (3, 2)

    graph = Topology(device)
    node = SpatialPoolerFlockNode(params=params)

    graph.add_node(node)

    input_block = MemoryBlock()
    input_block.tensor = torch.rand((params.flock_size,) + input_size, dtype=float_dtype, device=device)
    Connector.connect(input_block, node.inputs.sp.data_input)

    graph.prepare()

    node._unit.flock.cluster_centers = torch.tensor([[[1, 0, 0, 0, 0, 0],
                                                      [0, 1, 0, 0, 0, 0],
                                                      [0, 0, 0.5, 0.5, 0, 0],
                                                      [0, 0, 0.5, 0, 0.5, 0]],
                                                     [[1, 0, 0, 0, 0, 0],
                                                      [0, 1, 0, 0, 0, 0],
                                                      [0, 0, 1, 0, 0, 0],
                                                      [0, 0, 0, 1, 0, 0]]], dtype=float_dtype, device=device)

    data = torch.tensor([[0, 0, 1, 0],
                         [0.2, 0.3, 0.4, 0.1]], dtype=float_dtype, device=device)

    packet = InversePassOutputPacket(data, node.outputs.sp.forward_clusters)
    projected = node.recursive_inverse_projection_from_output(packet)

    # The result of the projection itself would be [[0, 0, 0.5, 0.5, 0, 0], ...], and it should be viewed as (2, 3, 2).
    expected_projection = torch.tensor([[[0, 0], [0.5, 0.5], [0, 0]],
                                        [[0.2, 0.3], [0.4, 0.1], [0, 0]]], dtype=float_dtype, device=device)

    assert same(expected_projection, projected[0].tensor)
