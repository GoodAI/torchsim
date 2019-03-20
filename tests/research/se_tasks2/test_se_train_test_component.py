from torchsim.core.eval2.measurement_manager import RunMeasurementManager
from torchsim.core.graph import Topology
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.se_tasks2.se_train_test_component import SETrainTestComponent, SETrainTestModel


class SeNodeGroupStub(SeNodeGroup):

    def __init__(self):
        super().__init__()
        self.testing = False
        self.use_se = False

    @property
    def is_se_testing_phase(self):
        return self.testing


class SEOrDatasetGraphStub(Topology, SETrainTestModel):
    testing: bool = False

    def __init__(self, device: str = 'cpu'):
        super().__init__(device)
        self.testing = False
        self.node_group = SeNodeGroupStub()

    def switch_to_training(self):
        self.testing = False

    def switch_to_testing(self):
        self.testing = True

    @property
    def se_node_group(self) -> SeNodeGroup:
        return self.node_group


def test_se_component():
    """Checks that the topology gets correctly switched after the node group reports the switch."""
    topology = SEOrDatasetGraphStub()
    run_measurement_manager = RunMeasurementManager(topology.name, {})
    component = SETrainTestComponent(topology, run_measurement_manager)

    component.after_topology_step()

    assert not component.is_in_testing_phase()
    assert component.is_in_training_phase()

    topology.node_group.testing = True

    component.after_topology_step()

    assert component.is_in_testing_phase()
    assert topology.testing

    topology.node_group.testing = False

    component.after_topology_step()

    assert not component.is_in_testing_phase()
    assert not topology.testing
