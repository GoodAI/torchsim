from abc import abstractmethod
from collections import OrderedDict

from typing import Dict

import torch

from torchsim.core.actions import AgentActionsDescriptor
from torchsim.core.datasets.space_divisor import SpaceDivisor
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.nodes.internals.actions_observable import ActionsObservable, ActionsDescriptorProvider
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.memory.tensor_creator import TensorCreator, TensorSurrogate
from torchsim.gui.observer_system import Observable, TextObservable
from torchsim.utils.sample_collection_overseer import SampleCollectionOverseer
from torchsim.utils.space_engineers_connector import SpaceEngineersConnector, SpaceEngineersConnectorConfig


class AbstractLocationObservable(TextObservable):

    _node: 'SpaceEngineersConnectorNode'

    def __init__(self, node: 'SpaceEngineersConnectorNode'):
        self._node = node

    @abstractmethod
    def get_location_tensor(self):
        pass

    def get_data(self):
        if self._node._unit is not None:
            tensor = self.get_location_tensor()
            if type(tensor) is not TensorSurrogate:
                return f"X,Y: {self.get_location_tensor()[0]:.2f}, {self.get_location_tensor()[1]:.2f}"
        return ""


class LocationObservable(AbstractLocationObservable):

    def get_location_tensor(self):
        return self._node._unit.task_to_agent_location


class LocationTargetObservable(AbstractLocationObservable):

    def get_location_tensor(self):
        return self._node._unit.task_to_agent_location_target


class TaskMetadataObservable(TextObservable):
    _node: 'SpaceEngineersConnectorNode'

    def __init__(self, node: 'SpaceEngineersConnectorNode'):
        self._node = node

    def get_data(self):
        if self._node._unit is not None:
            if type(self._node._unit.metadata_task_id) is not TensorSurrogate:
                return f"task id: {self._node._unit.metadata_task_id[0]:.0f}, " \
                       f"task instance: {self._node._unit.metadata_task_instance_id[0]:.0f},<br />" \
                       f"task status: {self._node._unit.metadata_task_status[0]:.0f}, " \
                       f"task instance status: {self._node._unit.metadata_task_instance_status[0]:.0f}, <br />" \
                       f"testing phase: {self._node._unit.metadata_testing_phase[0]:.0f}"
        return ""


class SpaceEngineersConnectorUnit(Unit):
    _connector: SpaceEngineersConnector
    observables: Dict[str, Observable]

    def __init__(self, creator: TensorCreator, actions_descriptor: AgentActionsDescriptor,
                 se_config: SpaceEngineersConnectorConfig, ip_address: str, port: int,
                 sample_collection_overseer: SampleCollectionOverseer):
        super().__init__(creator.device)

        self._connector = SpaceEngineersConnector(ip_address, port, actions_descriptor, se_config,
                                                  sample_collection_overseer)
        self._se_config = se_config
        self.env_img = creator.zeros((self._se_config.render_height,
                                      self._se_config.render_width, 3), device=self._device)
        self.reward = creator.zeros(1, device=self._device)
        label_size = self._se_config.task_to_agent_buffer_size
        self.task_to_agent_label = creator.zeros(label_size, device=self._device)
        self.task_to_agent_location = creator.zeros(self._se_config.LOCATION_SIZE, device=self._device)
        self.location_one_hot_horizontal_segments = 10
        self.location_one_hot_vertical_segments = 10
        location_one_hot_size = self.location_one_hot_vertical_segments * self.location_one_hot_horizontal_segments
        self.task_to_agent_location_int = creator.zeros(1, device=self._device)
        self.task_to_agent_location_one_hot = creator.zeros(location_one_hot_size, device=self._device)
        self.task_to_agent_label_target = creator.zeros(label_size, device=self._device)
        self.task_to_agent_location_target = creator.zeros(self._se_config.LOCATION_SIZE, device=self._device)
        self.task_to_agent_location_target_one_hot = creator.zeros(location_one_hot_size, device=self._device)
        self.metadata_task_id = creator.zeros(1, device=self._device)
        self.metadata_task_instance_id = creator.zeros(1, device=self._device)
        self.metadata_task_status = creator.zeros(1, device=self._device)
        self.metadata_task_instance_status = creator.zeros(1, device=self._device)
        self.metadata_testing_phase = creator.zeros(1, device=self._device)

        self.agent_to_task = creator.zeros(self._se_config.agent_to_task_buffer_size, device=self._device)

        self.run_n_se_steps_per_sim_step = 1  # mainly a debugging feature, prefer using skip_frames
        self._space_divisor = SpaceDivisor(self.location_one_hot_horizontal_segments,
                                           self.location_one_hot_vertical_segments, self._device)

    def step(self, action: torch.Tensor, agent_to_task_label: torch.Tensor, task_control: torch.Tensor):
        # Get the inputs as numpy
        action = action.cpu().numpy()
        agent_to_task_label = agent_to_task_label.cpu().numpy()
        task_control = task_control.cpu().numpy()

        img, reward, tta_label, tta_location, tta_label_target, tta_location_target, metadata = \
            None, None, None, None, None, None, None

        # Get the next step
        total_reward = 0
        for _ in range(0, self.run_n_se_steps_per_sim_step):
            img, reward, tta_label, tta_location, tta_label_target, tta_location_target, metadata = \
                self._connector.send_input_receive_output(action, agent_to_task_label, task_control)
            total_reward += reward
        reward = total_reward

        if self._se_config.reward_clipping_enabled:
            reward = max(-1, reward)
            reward = min(1, reward)

        # Set values of the output memory blocks
        tmp_img = torch.tensor(img, device=self._device)
        tmp_reward = torch.tensor([reward], device=self._device, dtype=torch.float32)
        tmp_task_to_agent_label = torch.tensor(tta_label, device=self._device)
        tmp_task_to_agent_location = torch.tensor(tta_location, device=self._device)
        tmp_task_to_agent_label_target = torch.tensor(tta_label_target, device=self._device)
        tmp_task_to_agent_location_target = torch.tensor(tta_location_target, device=self._device)
        tmp_metadata_task_id = torch.tensor([metadata[0]], device=self._device, dtype=torch.float32)
        tmp_metadata_task_instance_id = torch.tensor([metadata[1]], device=self._device, dtype=torch.float32)
        tmp_metadata_task_status = torch.tensor([metadata[2]], device=self._device, dtype=torch.float32)
        tmp_metadata_task_instance_status = torch.tensor([metadata[3]], device=self._device, dtype=torch.float32)
        tmp_metadata_testing_phase = torch.tensor([metadata[4]], device=self._device, dtype=torch.float32)

        self.env_img.copy_(tmp_img)
        self.reward.copy_(tmp_reward)
        self.task_to_agent_label.copy_(tmp_task_to_agent_label)
        self.task_to_agent_location.copy_(tmp_task_to_agent_location)
        self.task_to_agent_location_one_hot.copy_(
            self._space_divisor.get_landmark_normalize(tta_location[1], tta_location[0], -0.55, 9.55)[1])
        # TODO: why was the argmax over dim 1 and not zero?
        self.task_to_agent_location_int.copy_(torch.argmax(self.task_to_agent_location_one_hot, 0))
        # the maps provided by SE contain a 10x10 grid of cells; the center of the bottom left cell is 0, 0
        self.task_to_agent_label_target.copy_(tmp_task_to_agent_label_target)
        self.task_to_agent_location_target.copy_(tmp_task_to_agent_location_target)
        self.task_to_agent_location_target_one_hot.copy_(
            self._space_divisor.get_landmark_normalize(tta_location_target[1], tta_location_target[0], -0.55, 9.55)[1])
        self.metadata_task_id.copy_(tmp_metadata_task_id)
        self.metadata_task_instance_id.copy_(tmp_metadata_task_instance_id)
        self.metadata_task_status.copy_(tmp_metadata_task_status)
        self.metadata_task_instance_status.copy_(tmp_metadata_task_instance_status)
        self.metadata_testing_phase.copy_(tmp_metadata_testing_phase)

        return


class SpaceEngineersConnectorOutputs(MemoryBlocks):
    """All outputs sent by SE towards the agent.

    image_output: width X height RGB image
    reward_output: float (can be negative) indicating reward. From range -1,+1 for no skipped frames, but with
                   frame skipping can reach values in range e.g. -10,+10 (for skipframes=9)
    task_to_agent_label: label of an object provided by a task, mainly used in Task 0
    task_to_agent_location: location of an agent within the task grid, as X,Y
    task_to_agent_location_int: location of an agent within the task grid, as a unique number (X,Y -> Z)
    task_to_agent_location_one_hot: location of an agent within the task grid, as a unique number, converted to one-hot
    task_to_agent_target: label which the agent should be seing on its label input (not used by any task in milestone 1)
    task_to_agent_location_target: where the agent should go to, as X,Y
    task_to_agent_location_target_one_hot: same as above, but as a unique number (X,Y -> Z)
    metadata_task_id: id of the current task
    metadata_task_instance_id: id of the current instance of the current task
    metadata_task_status: status of the current or just finished task (0=nothing, -1=failed, 1=solved)
    metadata_task_instance_status: status of the current or just finished task instance (0=nothing, -1=failed, 1=solved)
    metadata_testing_phase: 0 when the task is sending training data, 1 when it is testing the agent
    """

    def __init__(self, owner):
        super().__init__(owner)
        self.image_output = self.create("Image output")
        self.reward_output = self.create("Reward output")
        self.task_to_agent_label = self.create("Task label output")
        self.task_to_agent_location = self.create("Task location output")
        self.task_to_agent_location_int = self.create("Task to agent location int")
        self.task_to_agent_location_one_hot = self.create("Task location one hot output")
        self.task_to_agent_target = self.create("Task label target output")
        self.task_to_agent_location_target = self.create("Task location target output")
        self.task_to_agent_location_target_one_hot = self.create("Task location target one hot output")
        self.metadata_task_id = self.create("Task id number output")
        self.metadata_task_instance_id = self.create("Task instance number output")
        self.metadata_task_status = self.create("Task status output")
        self.metadata_task_instance_status = self.create("Task instance status output")
        self.metadata_testing_phase = self.create("Testing phase info output")

    def prepare_slots(self, unit: SpaceEngineersConnectorUnit):
        self.image_output.tensor = unit.env_img
        self.reward_output.tensor = unit.reward
        self.task_to_agent_label.tensor = unit.task_to_agent_label
        self.task_to_agent_location.tensor = unit.task_to_agent_location
        self.task_to_agent_location_int.tensor = unit.task_to_agent_location_int
        self.task_to_agent_location_one_hot.tensor = unit.task_to_agent_location_one_hot
        self.task_to_agent_target.tensor = unit.task_to_agent_label_target
        self.task_to_agent_location_target.tensor = unit.task_to_agent_location_target
        self.task_to_agent_location_target_one_hot.tensor = unit.task_to_agent_location_target_one_hot
        self.metadata_task_id.tensor = unit.metadata_task_id
        self.metadata_task_instance_id.tensor = unit.metadata_task_instance_id
        self.metadata_task_status.tensor = unit.metadata_task_status
        self.metadata_task_instance_status.tensor = unit.metadata_task_instance_status
        self.metadata_testing_phase.tensor = unit.metadata_testing_phase


class SpaceEngineersConnectorInputs(Inputs):
    """All inputs SE requires from the agent. If not connected, they are filled with zeros.

    agent_action: a one-hot vector which indicates the action the agent wants to use this step
    agent_to_task_label: the label that the agent is assigning to the current image on its input (used in Task 0)
    task_control: special control flags that the agent can send to the task - the agent can reset a task or switch it
    """

    def __init__(self, owner):
        super().__init__(owner)
        self.agent_action = self.create("Agent action")
        self.agent_to_task_label = self.create("Task label input")
        self.task_control = self.create("Task control input")


class SpaceEngineersConnectorNode(WorkerNodeBase[SpaceEngineersConnectorInputs, SpaceEngineersConnectorOutputs], ActionsDescriptorProvider):
    """Node used for connecting to a running instance of Space Engineers.

    Use SpaceEngineersConnectorConfig for specifying how exactly you will connect.

    SampleCollectionOverseer is used only when converting the game's frames into a static dataset.

    Most tricky stuff (skip_frames): if your curriculum contains only a task 0 or 2000 (and optionally a task -1),
    the config will be modified to skip_frames = 0 and object_speed_multiplier = 10. This speeds up the movement of
    the objects in game, while disabling skipping of frames. The effect is that the agent sees the same frames as with
    skip_frames=9, but the game runs much faster.
    """

    inputs: SpaceEngineersConnectorInputs
    outputs: SpaceEngineersConnectorOutputs
    _observable_actions: ActionsObservable
    _unit: SpaceEngineersConnectorUnit

    def __init__(self, actions_descriptor: AgentActionsDescriptor, se_config: SpaceEngineersConnectorConfig,
                 ip_address: str='127.0.0.1', port: int=50000,
                 sample_collection_overseer: SampleCollectionOverseer = None):
        super().__init__("SpaceEngineersConnector", inputs=SpaceEngineersConnectorInputs(self),
                         outputs=SpaceEngineersConnectorOutputs(self))

        self._ip_address = ip_address
        self._port = port
        self._actions_descriptor = actions_descriptor
        self._se_config = se_config
        self._observable_actions = ActionsObservable(self)
        self._sample_collection_overseer = sample_collection_overseer

    def _create_unit(self, creator: TensorCreator):
        return SpaceEngineersConnectorUnit(creator, self._actions_descriptor, self._se_config, self._ip_address,
                                           self._port, self._sample_collection_overseer)

    def _get_observables(self) -> Dict[str, Observable]:
        observables = OrderedDict()
        observables['Task.Location'] = LocationObservable(self)
        observables['Task.Location target'] = LocationTargetObservable(self)
        observables['Task.Metadata'] = TaskMetadataObservable(self)
        return observables

    def _step(self):
        self._unit.step(self.inputs.agent_action.tensor, self.inputs.agent_to_task_label.tensor,
                        self.inputs.task_control.tensor)

    def get_actions_descriptor(self) -> AgentActionsDescriptor:
        return self._actions_descriptor
