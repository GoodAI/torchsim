from functools import partial
from typing import List

import torch
from torchsim.core.eval2.train_test_switchable import TrainTestSwitchable
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes import JoinNode, LambdaNode, ConstantNode
from torchsim.core.nodes.flock_networks.neural_network_flock import NeuralNetworkFlockParams, OutputActivation
from torchsim.core.nodes.network_flock_node import NetworkFlockNode, NetworkFlockNodeParams
from torchsim.core.nodes.squeeze_node import SqueezeNode
from torchsim.topologies.gradual_learning_topology import GradualLearningTopology, GateInputs, GateOutputs, PredictorInputs, \
    PredictorOutputs


def to_one_hot(length, inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
    r = torch.zeros(length)
    r[int(torch.round(inputs[0][0]).item())] = 1
    outputs[0].copy_(r)


class NNPredictorNodeGroup(NodeGroupBase[PredictorInputs, PredictorOutputs]):
    def __init__(self, num_predictors, input_data_size, name: str = "NNPredictorNodeGroup"):
        super().__init__(name, inputs=PredictorInputs(self), outputs=PredictorOutputs(self))

        p_node_params = NetworkFlockNodeParams()
        p_node_params.flock_size = num_predictors
        p_node_params.do_delay_coefficients = False
        p_node_params.do_delay_input = True
        p_node_params.normalize_coefficients = False
        p_node_params.negative_coefficients_removal = True
        p_node_params.buffer_size = 500
        p_node_params.batch_size = 400
        p_node_params.learning_period = 20

        p_network_params = NeuralNetworkFlockParams()
        p_network_params.flock_size = p_node_params.flock_size
        p_network_params.input_size = input_data_size
        p_network_params.hidden_size = input_data_size * 2
        p_network_params.output_size = input_data_size
        p_network_params.output_activation = OutputActivation.SOFTMAX
        p_network_params.learning_rate = 0.1
        p_network_params.coefficients_minimum_max = 0.3

        predictors = NetworkFlockNode(node_params=p_node_params, network_params=p_network_params, name="predictors")
        self.predictors = predictors
        self.add_node(predictors)

        to_one_hot_lambda_node = LambdaNode(partial(to_one_hot, input_data_size), 1, [(input_data_size,)])
        self.add_node(to_one_hot_lambda_node)

        Connector.connect(self.inputs.data.output, to_one_hot_lambda_node.inputs[0])
        Connector.connect(to_one_hot_lambda_node.outputs[0], predictors.inputs.input_data)
        Connector.connect(to_one_hot_lambda_node.outputs[0], predictors.inputs.target_data)
        # Connector.connect(self.inputs.prediction_target.output, predictors.inputs.target_data)
        Connector.connect(self.inputs.temporal_class_distribution.output, predictors.inputs.learning_coefficients)

        Connector.connect(predictors.outputs.prediction_output, self.outputs.predictors_activations.input)
        Connector.connect(predictors.outputs.error_output, self.outputs.predictors_activations_error.input)

    def switch_learning(self, on):
        self.predictors.is_learning_enabled = on


class NNGateNodeGroup(NodeGroupBase[GateInputs, GateOutputs]):
    def __init__(self, num_predictors, spatial_input_size, predicted_input_size, name: str = "NNGateNodeGroup"):
        super().__init__(name, inputs=GateInputs(self), outputs=GateOutputs(self))

        # join input and label
        inputs_join_node = JoinNode(flatten=True)
        self.add_node(inputs_join_node)

        # gate
        g_node_params = NetworkFlockNodeParams()
        g_node_params.flock_size = 1
        g_node_params.do_delay_coefficients = False
        g_node_params.do_delay_input = True
        g_node_params.learning_period = 20
        g_node_params.buffer_size = 1000
        g_node_params.batch_size = 900

        g_network_params = NeuralNetworkFlockParams()
        g_network_params.input_size = spatial_input_size + predicted_input_size
        g_network_params.mini_batch_size = 100
        g_network_params.hidden_size = g_network_params.input_size * 2
        g_network_params.output_size = num_predictors
        g_network_params.output_activation = OutputActivation.SOFTMAX
        g_network_params.learning_rate = 0.05

        gate = NetworkFlockNode(node_params=g_node_params, network_params=g_network_params, name="gate")
        self.gate = gate
        self.add_node(gate)

        # const
        learning_constant = ConstantNode([1], 1)
        self.add_node(learning_constant)

        # squeeze
        squeeze_node = SqueezeNode(0)
        self.add_node(squeeze_node)

        to_one_hot_lambda_node_p = LambdaNode(partial(to_one_hot, predicted_input_size), 1, [(predicted_input_size,)])
        self.add_node(to_one_hot_lambda_node_p)

        to_one_hot_lambda_node_s = LambdaNode(partial(to_one_hot, spatial_input_size), 1, [(spatial_input_size,)])
        self.add_node(to_one_hot_lambda_node_s)

        # connections
        Connector.connect(self.inputs.data.output, to_one_hot_lambda_node_s.inputs[0])
        Connector.connect(to_one_hot_lambda_node_s.outputs[0], inputs_join_node.inputs[0])
        Connector.connect(self.inputs.data_predicted.output, to_one_hot_lambda_node_p.inputs[0])
        Connector.connect(to_one_hot_lambda_node_p.outputs[0], inputs_join_node.inputs[1])

        Connector.connect(learning_constant.outputs.output, gate.inputs.learning_coefficients)
        Connector.connect(inputs_join_node.outputs.output, gate.inputs.input_data)
        Connector.connect(self.inputs.predictor_activations_target.output, gate.inputs.target_data)

        Connector.connect(gate.outputs.prediction_output, squeeze_node.inputs[0])
        Connector.connect(squeeze_node.outputs[0], self.outputs.gate_activations.input)

    def switch_learning(self, on):
        self.gate.is_learning_enabled = on


class GlNnTopology(GradualLearningTopology, TrainTestSwitchable):
    def get_predictors(self) -> NodeGroupBase[PredictorInputs, PredictorOutputs]:
        return self.predictors

    def get_gate(self) -> NodeGroupBase[GateInputs, GateOutputs]:
        return self.gate

    def get_num_predictors(self) -> int:
        return self.num_predictors

    def __init__(self, sequence_generators=None, num_predictors=2):
        self.num_predictors = num_predictors
        self.predictors = NNPredictorNodeGroup(self.num_predictors, 6)
        self.gate = NNGateNodeGroup(self.num_predictors, 4, 6)
        super().__init__(sequence_generators=sequence_generators, gamma=0.4)

    def switch_to_training(self):
        self.gate.switch_learning(True)
        self.predictors.switch_learning(True)

    def switch_to_testing(self):
        self.gate.switch_learning(False)
        self.predictors.switch_learning(False)


class FakeGateNodeGroup(NodeGroupBase[GateInputs, GateOutputs]):
    def __init__(self, num_predictors, name: str = "FakeGateNodeGroup"):
        super().__init__(name, inputs=GateInputs(self), outputs=GateOutputs(self))

        to_one_hot_lambda_node = LambdaNode(partial(to_one_hot, num_predictors), 1, [(num_predictors,)])
        self.add_node(to_one_hot_lambda_node)

        # connections
        Connector.connect(self.inputs.data.output, to_one_hot_lambda_node.inputs[0])

        Connector.connect(to_one_hot_lambda_node.outputs[0], self.outputs.gate_activations.input)


class GlFakeGateNnTopology(GradualLearningTopology):
    def get_predictors(self) -> NodeGroupBase[PredictorInputs, PredictorOutputs]:
        return self.predictors

    def get_gate(self) -> NodeGroupBase[GateInputs, GateOutputs]:
        return self.gate

    def get_num_predictors(self) -> int:
        return self.num_predictors

    def __init__(self, sequence_generators=None, num_predictors=2):
        self.num_predictors = num_predictors
        self.predictors = NNPredictorNodeGroup(self.num_predictors, 6)
        self.gate = FakeGateNodeGroup(self.num_predictors)
        super().__init__(sequence_generators=sequence_generators)

#
# class FakePredictorsNodeGroup(NodeGroupBase[PredictorInputs, PredictorOutputs]):
#     def __init__(self, num_predictors, name: str = "FakePredictorsNodeGroup"):
#         super().__init__(name, inputs=PredictorInputs(self), outputs=PredictorOutputs(self))
#
#
#
#
# class GlFakePredictorsNnTopology(GradualLearningTopology):
#     num_predictors = 2
#     gate = NNGateNodeGroup(num_predictors, 2)
#     predictors = FakePredictorsNodeGroup(num_predictors)
