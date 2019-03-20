import torch

from torchsim.core.eval.metrics.sp_convergence_metrics import average_sp_delta, average_boosting_duration, \
    num_boosted_clusters
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.graph.connection import Connector
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes import UnsqueezeNode, RandomNumberNode, SpatialPoolerFlockNode
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.topologies.mnist_sp_topology import MnistSpTopology


def test_sp_flock_node_accessor_types_and_dimensions():

    device = 'cuda'  # CPU not supported by SPFlock

    upper_bound = 107
    flock_input_size = upper_bound
    flock_size = 1

    num_cc = 21

    # define params
    sp_params = MnistSpTopology.get_sp_params(
        num_cluster_centers=num_cc,
        cluster_boost_threshold=1000,
        learning_rate=0.1,
        buffer_size=2 * 30,
        batch_size=30,
        input_size=flock_input_size,
        flock_size=flock_size,
        max_boost_time=1500
    )

    # random_node -> unsqueeze_node, sp_flock
    random_node = RandomNumberNode(upper_bound=upper_bound)
    unsqueeze_node = UnsqueezeNode(0)
    sp_node = SpatialPoolerFlockNode(sp_params.clone())

    Connector.connect(
        random_node.outputs.one_hot_output,
        unsqueeze_node.inputs.input)
    Connector.connect(
        unsqueeze_node.outputs.output,
        sp_node.inputs.sp.data_input)

    # update dimensions
    creator = AllocatingCreator(device=device)
    random_node.allocate_memory_blocks(creator)
    unsqueeze_node.allocate_memory_blocks(creator)
    sp_node.allocate_memory_blocks(creator)

    # make step
    random_node.step()
    unsqueeze_node.step()
    sp_node.step()

    # collect the results
    reconstruction = SpatialPoolerFlockNodeAccessor.get_reconstruction(sp_node)
    deltas = SpatialPoolerFlockNodeAccessor.get_sp_deltas(sp_node)
    boosting_durations = SpatialPoolerFlockNodeAccessor.get_sp_boosting_durations(sp_node)
    output_id = SpatialPoolerFlockNodeAccessor.get_output_id(sp_node)

    # check result properties
    assert type(reconstruction) is torch.Tensor
    assert type(deltas) is torch.Tensor
    assert type(boosting_durations) is torch.Tensor
    assert type(output_id) is int

    assert reconstruction.shape == (flock_size, flock_input_size)
    assert deltas.shape == (flock_size, num_cc, flock_input_size)
    assert boosting_durations.shape == (flock_size, num_cc)
    assert 0 <= output_id < num_cc

    # test the sp metrics
    delta = average_sp_delta(deltas)
    boosting_dur = average_boosting_duration(boosting_durations)

    nbc = num_boosted_clusters(boosting_durations)

    assert type(delta) is float
    assert type(boosting_dur) is float
    assert 0 <= boosting_dur <= 1000

    assert type(nbc) is float
    assert 0 <= nbc <= 1



