from typing import List, Optional

from eval_utils import observer_system_context, run_topology_with_ui
from torchsim.core.eval2.train_test_switchable import TrainTestSwitchable
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes.disentangled_world_node import create_default_temporal_classes
from torchsim.core.nodes.flock_networks.neural_network_flock import OutputActivation
from torchsim.core.physics_model.pymunk_physics import TemporalClass, Instance, InstanceShape, InstanceColor, PymunkParams
from torchsim.core.utils.sequence_generator import SequenceGenerator, diagonal_transition_matrix
from torchsim.research.research_topics.rt_4_3_1_gradual_world.node_groups.flock_network_group import FlockNetworkGroup
from torchsim.research.research_topics.rt_4_3_1_gradual_world.node_groups.gate_network_group import GateNetworkGroup
from torchsim.research.research_topics.rt_4_3_1_gradual_world.node_groups.switchable_world_group import SplittedWorldOutputs, \
    SwitchableWorldGroup, SwitchableWorldTopology
from torchsim.research.research_topics.rt_4_3_1_gradual_world.topologies.general_gl_topology import GradualLearningTopology, \
    PredictorGroupInputs, PredictorGroupOutputs, GateGroupOutputs, GateGroupInputs


class GlNnWorldTopology(GradualLearningTopology, TrainTestSwitchable):
    """Topology which implements parts of gradual learning by adaptive gating input to different networks"""

    def get_world(self) -> NodeGroupBase[EmptyInputs, SplittedWorldOutputs]:
        return self.world

    def get_predictors(self) -> NodeGroupBase[PredictorGroupInputs, PredictorGroupOutputs]:
        return self.predictors

    def get_gate(self) -> NodeGroupBase[GateGroupInputs, GateGroupOutputs]:
        return self.gate

    def get_num_predictors(self) -> int:
        return self.num_predictors

    def __init__(self,
                 temporal_class_definitions: Optional[List[List[TemporalClass]]] = None,
                 sx: int = 40,
                 sy: int = 100,
                 num_predictors: int = 2,
                 gamma: float = 0.5,
                 predicted_indexes: Optional[List[int]] = None,
                 gate_indexes: Optional[List[int]] = None,
                 predictor_activation: Optional[OutputActivation] = OutputActivation.IDENTITY,
                 is_gate_supervised: Optional[bool] = False,
                 flock_lr: Optional[float] = 0.1,
                 gate_lr: Optional[float] = 0.05,
                 coefficients_min_max: Optional[float] = 0.1,
                 predictor_hidden_s: Optional[int] = 20,
                 predictor_n_layers: Optional[int] = 1,
                 use_teleport: Optional[bool] = False
                 ):
        """
        Creates a topology of:
            -2 disentangled worlds that can be switched between
            -gate
            -network flock of predictors

        Args:
            temporal_class_definitions: configuration of the worlds
            sx: size of the world
            sy: size of the world
            num_predictors: number of predictors in the flock
            gamma: persistence of exponential decay applied to the predictor outputs (more the longer history considered)
            predicted_indexes: define subset of latent vector of the world to be predicted
            gate_indexes: subset of latent vector positions to be used for gating
            predictor_activation: activation on the output layer of the predictors (different data format possible)
            is_gate_supervised: input to the gate should be connected also as a target? (instead of argmin(avg(error))
        """
        super().__init__(gamma, is_gate_supervised)

        if temporal_class_definitions is None:
            # here, at least two worlds are necessary
            temporal_class_definitions = [
                create_default_temporal_classes(sx, sy),
                create_default_temporal_classes(sx, sy)
            ]

        self.num_predictors = num_predictors

        # define the world
        self.world = SwitchableWorldGroup(temporal_class_definitions=temporal_class_definitions,
                                          sx=sx,
                                          sy=sy,
                                          predictor_input_indexes=predicted_indexes,
                                          gate_indexes=gate_indexes,
                                          use_teleport=use_teleport)
        # define the flock
        self.predictors = FlockNetworkGroup(num_predictors=self.num_predictors,
                                            coefficients_minimum_max=coefficients_min_max,
                                            learning_rate=flock_lr,
                                            output_activation=predictor_activation,
                                            hidden_size=predictor_hidden_s,
                                            n_layers=predictor_n_layers)

        # define the gate
        self.gate = GateNetworkGroup(num_predictors=self.num_predictors,
                                     learning_rate=gate_lr)

        super().connect_topology()

    def switch_to_training(self):
        self.gate.switch_learning(True)
        self.predictors.switch_learning(True)

    def switch_to_testing(self):
        self.gate.switch_learning(False)
        self.predictors.switch_learning(False)


def define_sequence_generators() -> List[SequenceGenerator]:
    a = SequenceGenerator(
        [
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [5, 4, 3, 5, 4, 3, 5, 4, 3],
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [5, 4, 3, 5, 4, 3, 5, 4, 3],
        ]
        , diagonal_transition_matrix(4, 0.8))

    b = SequenceGenerator(
        [
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
        ]
        , diagonal_transition_matrix(2, 0.8))

    return [a, b]


def define_temporal_classes(sx: int, sy: int, three_worlds: bool) -> [List[List[TemporalClass]], List[int]]:
    """Get the temporal class definitions and indexes of data in the latent vector to be predicted"""

    pmp = PymunkParams()
    pmp.sx = sx
    pmp.sy = sy

    blue = Instance(pmp, 100, init_position=(40, 20),
                    init_direction=(1, 0),
                    color=InstanceColor.BLUE,
                    shape=InstanceShape.CIRCLE,
                    object_velocity=400,
                    rewrite_direction=True)

    red = Instance(pmp, 100,
                   init_direction=(-1, 0),
                   color=InstanceColor.RED,
                   shape=InstanceShape.SQUARE,
                   object_velocity=600,
                   rewrite_direction=True)

    green = Instance(pmp, 100,
                     init_direction=(-1, 0),
                     color=InstanceColor.GREEN,
                     shape=InstanceShape.TRIANGLE,
                     object_velocity=600,
                     rewrite_direction=True)

    # change between two objects of different appearance and speed
    temp_classes_w1 = \
        [
            # TemporalClass([
            #     Instance(100, init_position=(20, 20), init_velocity=(0, 20))
            # ]),
            TemporalClass([
                blue,
                red
            ])
        ]

    # not forgetting, just one object
    temp_classes_w2 = \
        [
            TemporalClass([
                blue
            ])
        ]

    if not three_worlds:
        return [temp_classes_w1, temp_classes_w2]

    # three objects
    temp_classes_w3 = \
        [
            TemporalClass([
                blue,
                red,
                green
            ])
        ]

    return [temp_classes_w1, temp_classes_w2, temp_classes_w3]


def setup_demo_model():
    """Configures the example model used for the demo of gradual learning at 15.3.2019

    It has:
        -a gate which receives 1-hot representation of color (length 3)
        -2 specialists (each supposed to learn predicting next X-coord: movement left/right)

    The graduality is tested using 3 different environments, they contain:
        -a) 2 objects (red one moving fast left; blue one moving slow right)
        -b) only a blue object
        -c) 3 objects (additional green one, which is behaving as the red one)

    The thing learn two objects in a), not forgetting shown on b) and learning new knowledge shown on c).

    Description of the demo available in the presentation here:
    https://docs.google.com/presentation/d/1YuUgWDtfRTeFeuypPcennT_heBLM5XaBGpBYhvPt_Bk/edit#slide=id.g5210ce2a68_13_10


    """

    sx = 40
    sy = 100

    three_worlds = True

    classes = define_temporal_classes(sx, sy, three_worlds=three_worlds)

    predicted_indexes = [0]  # [X, Y, dX, dY]
    if three_worlds:
        gate_indexes = [7, 8, 9]  # one-hot representation of shape(?) for 3 objects
    else:
        gate_indexes = [7, 8]  # one-hot representation for two objects (compatible with gate_supervised for 2 spec.)

    full_model: bool = True

    if full_model:
        model = GlNnWorldTopology(temporal_class_definitions=classes,
                                  predicted_indexes=predicted_indexes,  # or None
                                  gate_indexes=gate_indexes,  # or None
                                  sx=sx,
                                  sy=sy,
                                  num_predictors=2,
                                  gamma=0.8,  # was 0.5
                                  predictor_activation=OutputActivation.SIGMOID,
                                  is_gate_supervised=False,
                                  flock_lr=0.01,
                                  gate_lr=0.05,  # 0.005 worked, 0.001 is safer
                                  coefficients_min_max=0.3,  # 0.001 used
                                  predictor_hidden_s=30,
                                  predictor_n_layers=3,
                                  use_teleport=True)
    else:
        model = SwitchableWorldTopology(temporal_class_definitions=classes, sx=sx, sy=sy)

    return model


if __name__ == '__main__':

    # configure the experiment
    model = setup_demo_model()

    model.assign_ids()
    seed = 1123

    # Create simulation, it is registers itself to observer_system
    with observer_system_context('observers.yaml') as observer_system:
        run_topology_with_ui(topology=model,
                             seed=seed,
                             max_steps=0,
                             auto_start=False,
                             observer_system=observer_system)

    print('Running simulation, press enter to quit.')
    input()
