import random
from abc import abstractmethod, ABC
from typing import List, Optional

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.node_group import GroupOutputs, NodeGroupBase
from torchsim.core.nodes import SwitchNode
from torchsim.core.nodes.disentagled_world_renderer import DisentangledWorldRendererNode
from torchsim.core.nodes.disentangled_world_node import DisentangledWorldNodeParams, DisentangledWorldNode, \
    create_default_temporal_classes
from torchsim.core.nodes.switch_node import SwitchInputs
from torchsim.core.physics_model.pymunk_physics import TemporalClass, Instance, InstanceColor, PymunkParams
from torchsim.gui.observables import ObserverPropertiesItem
from torchsim.research.research_topics.rt_4_3_1_gradual_world.nodes.custom_nodes import create_predictions_gather_node


class SplittedWorldOutputs(GroupOutputs):

    def __init__(self, owner):
        super().__init__(owner=owner)

        self.predictor_inputs = self.create("Predictor Inputs")  # what is passed to the predictors
        # self.predictor_targets = self.create("Predictor Targets")  # what is being predicted by the predictors
        self.context = self.create("Context data")  # this is connected to the gate (anything useful for switching)


class SwitchableEnvironmentGroup(NodeGroupBase[EmptyInputs, SplittedWorldOutputs], ABC):
    """An input to the GL topology"""

    @abstractmethod
    def switch_input(self):
        pass

    @abstractmethod
    def switch_input_to(self, idx):
        pass


class SwitchableWorldGroup(SwitchableEnvironmentGroup):

    def teleport(self, instances: List[Instance]):
        """Experimental version of the teleport, works only in X for width of the map of 100

        Teleports objects on the edges of the environment so that they do not bounce
        """

        eps = 7

        for inst in instances:
            color = inst.color
            new_color = color

            while new_color == color:
                new_color = random.choice(list(InstanceColor))

            # inst.color = new_color
            # inst.pm_body.position = (10, 10)
            # inst.pm_body.position = inst.init_position
            pos = inst.pm_body.position

            if abs(0 - pos[0]) < eps:
                pos[0] = 100 - 10
            if abs(100 - pos[0]) < eps:
                pos[0] = 0 + 10

            inst.pm_body.position = pos

            inst.pm_body.velocity = tuple([inst.object_velocity * direct for direct in list(inst.init_direction)])

    def __init__(self,
                 temporal_class_definitions: List[List[TemporalClass]],
                 sx: int,
                 sy: int,
                 predictor_input_indexes: Optional[List[int]] = None,
                 # predictor_target_indexes: Optional[List[int]] = None,
                 gate_indexes: Optional[List[int]] = None,
                 add_rendering: bool = True,
                 name: str = "Switchable Worlds",
                 use_teleport: Optional[bool] = False):
        """ Creates a list of DisentangledWorld instances between whose outputs you can switch.

        It is also supposed to split the description of the world into Oututs.predicted_data
        (e.g. position of one object) and Outputs.context part (input to the gate)

        Args:
            temporal_class_definitions: for each world, there should be a list of temporal classes (objects in the world)
            (currently just one object per world supported/tested in the topology)
            predictor_input_indexes: indicate positions in the latent vector which you want to pass to predictors
            gate_indexes: the same as predicted_indexes, if None, the gate receives entire latent vector
            sx: size of the world
            sy: size of the world
            add_rendering: bool, add the WorldRendered nodes for each node?
            name:
        """

        super().__init__(name, outputs=SplittedWorldOutputs(self))

        self.validate_params(temporal_class_definitions)

        self.num_worlds = len(temporal_class_definitions)

        switch_node_predicted = SwitchNode(self.num_worlds)
        self.switch_node = switch_node_predicted
        self.add_node(switch_node_predicted)

        switch_node_context = SwitchNode(self.num_worlds)
        self.switch_node_seq = switch_node_context
        self.add_node(switch_node_context)

        if use_teleport:
            teleport = self.teleport
        else:
            teleport = None

        self.env_nodes = []
        for i, temporal_class_def in enumerate(temporal_class_definitions):
            params = DisentangledWorldNodeParams(sx=sx, sy=sy,
                                                 temporal_classes=temporal_class_def,
                                                 use_pygame=False)
            env_node = DisentangledWorldNode(params=params, name=f"World {i}", post_collision_callback=teleport)
            self.add_node(env_node)
            self.env_nodes.append(env_node)

            if add_rendering:
                renderer = DisentangledWorldRendererNode(params=params, name=f"World {i} render")
                self.add_node(renderer)
                Connector.connect(env_node.outputs.latent, renderer.inputs.latent)

            # latent_vector -> gate
            self._connect_world_to_subnetwork(gate_indexes, env_node, switch_node_context.inputs, i,
                                              "Gather gate inputs")
            # latent_vector -> predictors
            self._connect_world_to_subnetwork(predictor_input_indexes, env_node, switch_node_predicted.inputs, i,
                                              "Gather predicted inputs")

        Connector.connect(switch_node_predicted.outputs.output, self.outputs.predictor_inputs.input)
        Connector.connect(switch_node_context.outputs.output, self.outputs.context.input)

    def _connect_world_to_subnetwork(self,
                                     vector_indexes: List[int],
                                     env_node: DisentangledWorldNode,
                                     switch_node_inputs: SwitchInputs,
                                     index: int,
                                     name: str):
        """Connect the latent vector to the network (either gate or predictor)

        The vector_indexes can be either None or list of indexes to be taken from the latent vector.

        If the vector_indexes is not None, the latent vector will be connected directly to the sub-network,
        otherwise the gather node will be inserted between them.
        """
        # send the entire input?
        if vector_indexes is None:
            Connector.connect(env_node.outputs.latent, switch_node_inputs[index])
        else:
            gather_input = create_predictions_gather_node(vector_indexes,
                                                          num_objects=1,
                                                          name=f"{name} - {index}")
            self.add_node(gather_input)
            Connector.connect(env_node.outputs.latent, gather_input.inputs[0])
            Connector.connect(gather_input.outputs[0], switch_node_inputs[index])

    def switch_input(self):
        idx = (self.switch_node.active_input_index + 1) % self.num_worlds
        self.switch_input_to(idx)

    def switch_input_to(self, idx):
        for env_node in self.env_nodes:
            env_node.skip_execution = True
        self.env_nodes[idx].skip_execution = False
        self.switch_node.change_input(idx)
        self.switch_node_seq.change_input(idx)

    @staticmethod
    def validate_params(temporal_class_definitions: List[List[TemporalClass]]):
        """It is required that all the worlds have the same dimension of the output"""

        num_objects = len(temporal_class_definitions[0])

        for temp_class_def in temporal_class_definitions:
            assert len(temp_class_def) == num_objects

        # currently, the thing is not tested for num_objects > 1
        assert num_objects == 1


class SwitchableWorldTopology(Topology):
    """Topology used just for debugging purposes (test how the inputs can be swtiched manually)"""

    def __init__(self,
                 temporal_class_definitions: Optional[List[List[TemporalClass]]] = None,
                 sx: int = 40,
                 sy: int = 100,
                 device: str = 'cuda'):
        super().__init__(device)

        if temporal_class_definitions is None:
            # here, at least two worlds are necessary
            temporal_class_definitions = [
                create_default_temporal_classes(sx, sy),
                create_default_temporal_classes(sx, sy)
            ]

        self.world = SwitchableWorldGroup(temporal_class_definitions, sx, sy)
        self.add_node(self.world)

    def get_properties(self) -> List[ObserverPropertiesItem]:
        props = super().get_properties()
        return props + [
            self._prop_builder.button("switch input", self.switch_input),
        ]

    def switch_input(self):
        self.world.switch_input()
