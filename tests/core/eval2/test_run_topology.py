from eval_utils import run_topology_with_ui, run_topology
from torchsim.core.graph import Topology
from torchsim.core.nodes import LambdaNode
from torchsim.gui.observer_system import ObserverSystem
from torchsim.gui.observer_system_void import ObserverSystemVoid


class TopologyStub(Topology):
    def __init__(self, step_callback):
        super().__init__('cpu')

        self.add_node(LambdaNode(lambda inputs, outputs: step_callback(), 0, []))


class TestBasicExperimentTemplate:
    def test_run_topology(self):
        def callback():
            callback.called = True

        callback.called = False
        ObserverSystem.initialized = False

        run_topology(TopologyStub(callback), max_steps=1, save_model_after_run=False)

        assert callback.called

    def test_run_topology_in_ui(self):
        def callback():
            callback.called = True

        callback.called = False
        ObserverSystem.initialized = False
        observer_system = ObserverSystemVoid(None)

        run_topology_with_ui(TopologyStub(callback), max_steps=1, observer_system=observer_system)

        assert callback.called
