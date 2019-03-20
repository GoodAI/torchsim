from abc import ABC, abstractmethod
from typing import List, Any

import torch
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import GroupInputs, GroupOutputs, NodeGroupBase
from torchsim.core.nodes import LambdaNode
from torchsim.core.utils.sequence_generator import SequenceGenerator, diagonal_transition_matrix
from torchsim.core.utils.tensor_utils import id_to_one_hot
from torchsim.gui.observables import ObserverPropertiesItem
from torchsim.significant_nodes.switchable_sequence_nodegroup import SwitchableSequencesNodeGroup


class PredictorInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.data = self.create("Data")
        self.temporal_class_distribution = self.create("Temporal class distribution")


class PredictorOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.predictors_activations = self.create("Predictors activations")
        self.predictors_activations_error = self.create("Predictors activations error")


class GateInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.data = self.create("Data (y labels)")
        self.data_predicted = self.create("Data (x predicted)")
        self.predictor_activations_target = self.create("Predictor activations target")


class GateOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.gate_activations = self.create("Predictor activations")


class GradualLearningTopology(Topology, ABC):
    def __init__(self, sequence_generators: List[Any] = None, gamma=0.8):
        super().__init__('cpu')

        # environment
        if sequence_generators is None:
            sequence_generator_0 = SequenceGenerator(
                [
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [5, 4, 3, 5, 4, 3, 5, 4, 3],
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [5, 4, 3, 5, 4, 3, 5, 4, 3],
                ]
                , diagonal_transition_matrix(4, 0.8))
            sequence_generator_1 = SequenceGenerator(
                [
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [5, 4, 3, 5, 4, 3, 5, 4, 3],
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [5, 4, 3, 5, 4, 3, 5, 4, 3],
                ]
                , diagonal_transition_matrix(4, 0.8))
            sequence_generators = [sequence_generator_0, sequence_generator_1]

        self.sequences_len = len(sequence_generators)

        # switch
        switch_node = SwitchableSequencesNodeGroup(sequence_generators)
        self.switch_node = switch_node
        self.add_node(switch_node)

        # gate
        self.add_node(self.get_gate())

        # predictors
        self.add_node(self.get_predictors())

        # discount buffer
        def discount(inputs: List[torch.Tensor], outputs: List[torch.Tensor], memory: List[torch.Tensor]):
            cum_val = (1. - gamma) * inputs[0].squeeze() + gamma * memory[0]
            memory[0].copy_(cum_val)
            outputs[0].copy_(cum_val)
        discount_node = LambdaNode(discount, 1, [(self.get_num_predictors(),)],
                                   memory_shapes=[(self.get_num_predictors(),)],
                                   name="Discount")
        self.add_node(discount_node)

        # argmin
        def argmin(inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
            outputs[0].copy_(id_to_one_hot(inputs[0].argmin(), self.get_num_predictors()))
        argmin_lambda_node = LambdaNode(argmin, 1, [(self.get_num_predictors(),)], name="Argmin")
        self.add_node(argmin_lambda_node)

        # dot
        def dot_product(inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
            outputs[0].copy_(inputs[0].squeeze().dot(inputs[1].squeeze()))
        dot_node = LambdaNode(dot_product, 2, [(1,)], name="predictors error dot gate activations")
        self.dot_node = dot_node
        self.add_node(dot_node)

        # connections
        Connector.connect(switch_node.outputs.output, self.get_predictors().inputs.data)

        Connector.connect(switch_node.outputs.output, self.get_gate().inputs.data_predicted)
        Connector.connect(switch_node.outputs.sequence_num, self.get_gate().inputs.data)

        Connector.connect(self.get_predictors().outputs.predictors_activations_error, discount_node.inputs[0])
        Connector.connect(discount_node.outputs[0], argmin_lambda_node.inputs[0])
        Connector.connect(argmin_lambda_node.outputs[0], self.get_gate().inputs.predictor_activations_target)

        Connector.connect(self.get_gate().outputs.gate_activations,
                          self.get_predictors().inputs.temporal_class_distribution,
                          is_backward=True)

        Connector.connect(self.get_predictors().outputs.predictors_activations_error, dot_node.inputs[0])
        Connector.connect(self.get_gate().outputs.gate_activations, dot_node.inputs[1])

    def get_properties(self) -> List[ObserverPropertiesItem]:
        props = super().get_properties()
        return props + [
            self._prop_builder.collapsible_header(f'Experiment', True),
            self._prop_builder.button("switch input", self.switch_input),
        ]

    def switch_input(self):
        self.switch_node.switch_input()

    def switch_input_to(self, idx):
        self.switch_node.switch_input_to(idx)

    def get_error(self):
        return self.dot_node.outputs[0].tensor.item()

    @abstractmethod
    def get_predictors(self) -> NodeGroupBase[PredictorInputs, PredictorOutputs]:
        pass

    @abstractmethod
    def get_gate(self) -> NodeGroupBase[GateInputs, GateOutputs]:
        pass

    @abstractmethod
    def get_num_predictors(self) -> int:
        pass
