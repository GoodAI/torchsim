import logging
import tempfile

import pytest

from torchsim.core.graph import Topology
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.topologies.mnist_sp_topology import MnistSpTopology
from tests.core.graph.test_save_load import TAState
from tests.testing_utils import discover_main_topology_classes

logger = logging.getLogger(__name__)

skip_topologies = ['BaselinesReinforcementLearningTopology', 'SeToyArchDebugTopology',
                   'SEDatasetSampleCollectionTopology', 'DebugAgentTopology', 'NNetTopology',
                   'LrfObjectDetectionTopology']


@pytest.mark.parametrize('topology_class', discover_main_topology_classes())
def test_topologies_can_be_initialized(topology_class):
    topology_class()


@pytest.mark.slow
@pytest.mark.parametrize('topology_class', discover_main_topology_classes(skip_topologies))
def test_topologies_can_run_step(topology_class):
    """Just try that topologies can be initialized and run for one step."""
    topology_class().step()


@pytest.mark.parametrize('topology_class', [MnistSpTopology])#, Task0TaSeTopology, L3ConvTopology, L3SpConvTopology, GradualLearningBasicTopology])
def test_save_load(topology_class):
    """Unit test for saving and loading of topologies.

    1. Initialize a topology
    2. Take a snapshot of its state ("before state")
    3. Save the topology
    4. Change it
    5. Verify that it changed from the before state
    6. Load the topology
    7. Verify that it is now equal to the before state
    """
    topology = topology_class()
    topology.step()
    state_before = TAState(topology)

    with tempfile.TemporaryDirectory() as directory:
        saver = Saver(directory)
        topology.save(saver)
        saver.save()

        change_topology(topology)
        state_after_change = TAState(topology)
        assert state_after_change != state_before

        loader = Loader(directory)
        topology.load(loader)

    state_after_load = TAState(topology)
    assert state_before == state_after_load


@pytest.mark.skip(reason="Specialized test")
@pytest.mark.parametrize('topology_class, n_steps', [(MnistSpTopology, 1000)])
def test_save_load_many_steps(topology_class, n_steps):
    """A variation of the save/load test, for multi-step state changes.

    There can be parts of a node's state (e.g. a random number that was generated) that is not stored immediately in
    memory blocks but manifests themselves in the memory blocks after a number of steps.

    This is test specialized in detecting the failure to save/load those parts of the state.
    """
    topology = topology_class()
    topology.step()
    state_before = TAState(topology)

    with tempfile.TemporaryDirectory() as directory:
        saver = Saver(directory)
        topology.save(saver)
        saver.save()

        for _ in range(n_steps):
            topology.step()
        state_after_steps = TAState(topology)
        assert state_after_steps != state_before

        loader = Loader(directory)
        topology.load(loader)

    state_after_load = TAState(topology)
    assert state_after_load == state_before

    for _ in range(n_steps):
        topology.step()
    state_after_load_and_steps = TAState(topology)
    assert state_after_load_and_steps == state_after_steps


def change_topology(topology: Topology):
    """Changes the state of the topology.

    Just running a step is not guaranteed to effect a change, so we explicitly alter the memory blocks.
    """
    for ta_node in TAState.get_ta_nodes(topology):
        for block in ta_node.memory_blocks:
            if block is not None and block.tensor is not None:
                block.tensor.add_(1)
