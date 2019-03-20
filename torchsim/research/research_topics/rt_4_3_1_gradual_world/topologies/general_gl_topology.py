from abc import ABC, abstractmethod
from typing import List

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import GroupInputs, GroupOutputs, NodeGroupBase
from torchsim.gui.observables import ObserverPropertiesItem
from torchsim.research.research_topics.rt_4_3_1_gradual_world.node_groups.switchable_world_group import \
    SwitchableEnvironmentGroup
from torchsim.research.research_topics.rt_4_3_1_gradual_world.nodes.custom_nodes import create_discount_node, \
    create_arg_min_node, create_dot_product_node, create_delay_node


class PredictorGroupInputs(GroupInputs):

    def __init__(self, owner):
        super().__init__(owner)

        self.data = self.create("Data")  # shared as both inputs and targets for the predictors
        self.learning_coefficients = self.create("Learning Coefficients")  # how much should each predictor learn this


class PredictorGroupOutputs(GroupOutputs):

    def __init__(self, owner):
        super().__init__(owner)

        self.predictors_activations = self.create("Predictors activations")
        self.predictors_activation_errors = self.create("Predictors activation errors")


class GateGroupInputs(GroupInputs):

    def __init__(self, owner):
        super().__init__(owner)

        self.data = self.create("Input Data")  # input to the gate
        self.targets = self.create("Gate Targets")  # usually argmin of predictor errors


class GateGroupOutputs(GroupOutputs):

    def __init__(self, owner):
        super().__init__(owner)

        self.outputs = self.create("Gating output")  # used for gating outputs of the predictors


class GradualLearningTopology(Topology, ABC):

    _gamma: float
    _is_gate_supervised: bool

    def __init__(self,
                 gamma: float,
                 is_gate_supervised: bool):
        super().__init__('cpu')
        self._gamma = gamma
        self._is_gate_supervised = is_gate_supervised

    def connect_topology(self):
        """Wires the topology together"""

        # world
        self.add_node(self.get_world())

        # gate
        self.add_node(self.get_gate())

        # predictors
        self.add_node(self.get_predictors())

        # compute the argmin of the discounted average error of each predictor
        discount_node = create_discount_node(self._gamma, self.get_num_predictors())
        self.add_node(discount_node)
        argmin_lambda_node = create_arg_min_node(self.get_num_predictors())
        self.add_node(argmin_lambda_node)

        # just for debugging purposes, should be minimized
        self.dot_node = create_dot_product_node(input_size=self.get_num_predictors(), output_sizes=[(1,)],
                                                name="predictors error dot gate activations")
        self.add_node(self.dot_node)

        # delay the output of the gate
        self.delay_node = create_delay_node(num_predictors=self.get_num_predictors())
        self.add_node(self.delay_node)

        # world -> networks
        Connector.connect(self.get_world().outputs.predictor_inputs, self.get_predictors().inputs.data)
        Connector.connect(self.get_world().outputs.context, self.get_gate().inputs.data)

        # predictors -> argmin(avg(error))
        Connector.connect(self.get_predictors().outputs.predictors_activation_errors, discount_node.inputs[0])
        Connector.connect(discount_node.outputs[0], argmin_lambda_node.inputs[0])

        if self._is_gate_supervised:
            # input to the gate -> gate target
            # (expects the target is of one-hot format of size [num_predicotrs] and reasonably gates the data)
            Connector.connect(self.get_world().outputs.context, self.get_gate().inputs.targets)
        else:
            # argmin(avg(error)) -> gate target
            Connector.connect(argmin_lambda_node.outputs[0], self.get_gate().inputs.targets)

        # gate -> learning coefficients
        Connector.connect(self.get_gate().outputs.outputs,
                          self.get_predictors().inputs.learning_coefficients, is_backward=True)
        Connector.connect(self.get_gate().outputs.outputs, self.delay_node.inputs[0])

        # delayed gate -> product (debug) <- predictors
        Connector.connect(self.get_predictors().outputs.predictors_activation_errors, self.dot_node.inputs[0])
        Connector.connect(self.delay_node.outputs[0], self.dot_node.inputs[1])

    def get_properties(self) -> List[ObserverPropertiesItem]:
        props = super().get_properties()
        return props + [
            self._prop_builder.collapsible_header(f'Experiment', True),
            self._prop_builder.button("switch input", self.switch_input),
        ]

    def switch_input(self):
        self.get_world().switch_input()

    def switch_input_to(self, idx):
        self.get_world().switch_input_to(idx)

    def get_error(self):
        return self.dot_node.outputs[0].tensor.item()

    @abstractmethod
    def get_predictors(self) -> NodeGroupBase[PredictorGroupInputs, PredictorGroupOutputs]:
        pass

    @abstractmethod
    def get_gate(self) -> NodeGroupBase[GateGroupInputs, GateGroupOutputs]:
        pass

    @abstractmethod
    def get_world(self) -> SwitchableEnvironmentGroup:
        pass

    @abstractmethod
    def get_num_predictors(self) -> int:
        pass
