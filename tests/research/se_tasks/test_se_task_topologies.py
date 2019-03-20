from typing import Set

import pytest

from torchsim.core import FLOAT_NEG_INF
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval.node_accessors.se_io_accessor import SeIoAccessor
from torchsim.core.nodes.dataset_se_objects_node import DatasetSeObjectsParams, DatasetSeObjectsUnit
from torchsim.research.se_tasks.topologies.se_task0_basic_topology import SeT0BasicTopology
from torchsim.research.se_tasks.topologies.se_task0_convolutionalSP_topology import SeT0ConvSPTopology
from torchsim.research.se_tasks.topologies.se_task0_convolutional_expert_topology import SeT0ConvTopology
from torchsim.research.se_tasks.topologies.se_task0_narrow_hierarchy import SeT0NarrowHierarchy
from torchsim.research.se_tasks.topologies.se_task0_topology import SeT0TopologicalGraph


def inheritors(base_class):
    subclasses = set()
    work = [base_class]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def get_task_0_topologies() -> Set[SeT0TopologicalGraph]:
    return inheritors(SeT0TopologicalGraph)


def test_get_task_0_topologies():
    topologies = get_task_0_topologies()

    assert SeT0BasicTopology in topologies
    assert SeT0NarrowHierarchy in topologies
    assert SeT0ConvSPTopology in topologies
    assert SeT0ConvTopology in topologies
    assert SeT0TopologicalGraph not in topologies


@pytest.mark.parametrize('topology_class', get_task_0_topologies())
def test_se_task0_topologies_init(topology_class):
    """Test that all topologies which are supposed to solve SE task 0 can be initialized."""
    topology_class()


@pytest.mark.slow
@pytest.mark.parametrize('topology_class', get_task_0_topologies())
def test_se_task0_topologies_step(topology_class):
    """Test one step of all topologies which are supposed to solve SE task 0."""

    params = DatasetSeObjectsParams(save_gpu_memory=True)
    params.dataset_size = SeDatasetSize.SIZE_24

    topology = topology_class(use_dataset=True)

    topology.order_nodes()
    topology._update_memory_blocks()
    topology.step()

    landmark_id = SeIoAccessor.get_landmark_id_int(topology.se_io.outputs)
    label_id = SeIoAccessor.get_label_id(topology.se_io)

    assert type(landmark_id) is float
    assert landmark_id == FLOAT_NEG_INF

    assert type(label_id) is int
    assert label_id <= SeIoAccessor.get_num_labels(topology.se_io)

    assert SeIoAccessor.get_num_labels(topology.se_io) == DatasetSeObjectsUnit.NUM_LABELS


