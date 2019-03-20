import torch

from torchsim.core.actions import AgentActionsDescriptor
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.nodes.internals.grid_world import GridWorld, GridWorldInputs, GridWorldOutputs, GridWorldParams
from torchsim.core.nodes.internals.actions_observable import ActionsDescriptorProvider, ActionsObservable
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver


class GridWorldNode(WorkerNodeBase[GridWorldInputs, GridWorldOutputs], ActionsDescriptorProvider):
    """Simple maze world generating the map and the moving character.

    The movement of the character is controlled by GridWorldInputs.
    Various aspects of the world including the maps and other properties are maintained by the GridWorldParams.
    """

    _unit: GridWorld
    inputs: GridWorldInputs
    outputs: GridWorldOutputs
    _observable_actions: ActionsObservable

    def __init__(self, params: GridWorldParams, name="GridWorld"):
        super().__init__(name=name, inputs=GridWorldInputs(self),
                         outputs=GridWorldOutputs(self))
        self._params = params.clone()
        self._actions_descriptor = AgentActionsDescriptor()

    def _create_unit(self, creator: TensorCreator):
        return GridWorld(creator, self._params.clone())

    def _step(self):
        self._unit.step(self.inputs.agent_action.tensor)

    def get_actions_descriptor(self) -> AgentActionsDescriptor:
        return self._actions_descriptor

    def clone(self) -> 'GridWorldNode':
        return GridWorldNode(self._params, self.name)


class MultiGridWorld(Unit):
    def __init__(self, creator: TensorCreator, n_worlds: int, params: GridWorldParams):
        super().__init__(creator.device)
        self._params = params
        self._n_worlds = n_worlds

        self._units = [GridWorld(creator, params.clone()) for _ in range(self._n_worlds)]

        first_unit = self._units[0]

        def stacked(tensor):
            size = [self._n_worlds] + list(tensor.shape)
            return creator.zeros(size, dtype=tensor.dtype, device=tensor.device)

        self.last_images = stacked(first_unit.last_image)
        self.egocentric_images = stacked(first_unit.egocentric_image)
        self.last_positions = stacked(first_unit.last_position)
        self.last_actions = stacked(first_unit.last_action)
        self.last_position_one_hot_matrices = stacked(first_unit.last_position_one_hot_matrix)
        self.rewards = stacked(first_unit.reward)
        self.bit_map_last_actions = stacked(first_unit.bit_map_last_action)
        self.ego_last_actions = stacked(first_unit.ego_last_action)

    def step(self, agent_actions: torch.Tensor):
        individual_actions = agent_actions.unbind(dim=0)
        for unit, individual_action in zip(self._units, individual_actions):
            unit.step(individual_action)

        torch.stack([unit.last_image for unit in self._units], dim=0, out=self.last_images)
        torch.stack([unit.egocentric_image for unit in self._units], dim=0, out=self.egocentric_images)
        torch.stack([unit.last_position for unit in self._units], dim=0, out=self.last_positions)
        torch.stack([unit.last_action for unit in self._units], dim=0, out=self.last_actions)
        torch.stack([unit.last_position_one_hot_matrix for unit in self._units], dim=0,
                    out=self.last_position_one_hot_matrices)
        torch.stack([unit.reward for unit in self._units], dim=0, out=self.rewards)
        torch.stack([unit.bit_map_last_action for unit in self._units], dim=0, out=self.bit_map_last_actions)
        torch.stack([unit.ego_last_action for unit in self._units], dim=0, out=self.ego_last_actions)

    def _save(self, saver: Saver):
        super()._save(saver)

        for i, unit in enumerate(self._units):
            unit.save(saver.create_child(f'sub_unit_{i}'))

    def _load(self, loader: Loader):
        super()._load(loader)

        for i, unit in enumerate(self._units):
            unit.load(loader.load_child(f'sub_unit_{i}'))


class MultiGridWorldOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output_images = self.create("Output_images")
        self.egocentric_images = self.create("Egocentric_images")
        self.output_positions = self.create("Output_positions")
        self.output_actions = self.create("Output_actions")
        self.output_pos_one_hot_matrices = self.create("Output_pos_one_hot_matrices")
        self.rewards = self.create("Rewards")
        self.output_image_actions = self.create("Output_image_actions")
        self.egocentric_image_actions = self.create("Egocentric_image_actions")

    def prepare_slots(self, unit: MultiGridWorld):
        self.output_images.tensor = unit.last_images
        self.egocentric_images.tensor = unit.egocentric_images
        self.output_positions.tensor = unit.last_positions
        self.output_actions.tensor = unit.last_actions
        self.output_pos_one_hot_matrices.tensor = unit.last_position_one_hot_matrices
        self.rewards.tensor = unit.rewards
        self.output_image_actions.tensor = unit.bit_map_last_actions
        self.egocentric_image_actions.tensor = unit.ego_last_actions


class MultiGridWorldInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.agent_actions = self.create("Actions")


class MultiGridWorldNode(WorkerNodeBase[MultiGridWorldInputs, MultiGridWorldOutputs], ActionsDescriptorProvider):
    """Multiple grid world instances in one node."""

    _unit: MultiGridWorld
    inputs: MultiGridWorldInputs
    outputs: MultiGridWorldOutputs
    _observable_actions: ActionsObservable

    def __init__(self, n_worlds: int, params: GridWorldParams, name="MultiGridWorld"):
        super().__init__(name=name, inputs=MultiGridWorldInputs(self),
                         outputs=MultiGridWorldOutputs(self))
        self._params = params.clone()
        self._actions_descriptor = AgentActionsDescriptor()
        self._n_worlds = n_worlds

    def _create_unit(self, creator: TensorCreator):
        return MultiGridWorld(creator, self._n_worlds, self._params.clone())

    def _step(self):
        self._unit.step(self.inputs.agent_actions.tensor)

    def get_actions_descriptor(self) -> AgentActionsDescriptor:
        return self._actions_descriptor
