import numpy as np
from typing import List

import torch
from functools import partial

from torchsim.core.actions import AgentActionsDescriptor
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.gui.observer_system import ObserverPropertiesItem


class ActionMonitor(Unit):

    def __init__(self, creator: TensorCreator, actions_descriptor: AgentActionsDescriptor):
        super().__init__(creator.device)
        self.action_output = creator.zeros(len(actions_descriptor.action_names()), dtype=self._float_dtype, device=self._device)

    def step(self, input_action, override_checked, actions_values):
        input_action = input_action.cpu().numpy()

        if override_checked:
            input_action = np.array([1 if action else 0 for action in actions_values])

        inp_tensor = torch.tensor(input_action, dtype=self._float_dtype, device=self._device)
        self.action_output.copy_(inp_tensor)


class ActionMonitorInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.action_in = self.create("Action in")


class ActionMonitorOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.action_out = self.create("Action out")

    def prepare_slots(self, unit: ActionMonitor):
        self.action_out.tensor = unit.action_output


class ActionMonitorNode(WorkerNodeBase[ActionMonitorInputs, ActionMonitorOutputs]):
    """Node which reads (and can optionally override) the actions of the agent in Space Engineers."""

    _unit: ActionMonitor
    inputs: ActionMonitorInputs
    outputs: ActionMonitorOutputs
    _override_checked = False

    def __init__(self, actions_descriptor: AgentActionsDescriptor, name="Action Monitor"):
        super().__init__(name=name, inputs=ActionMonitorInputs(self), outputs=ActionMonitorOutputs(self))
        self._actions_descriptor = actions_descriptor

        self._actions_values = [False for i in actions_descriptor.action_names()]

    def _create_unit(self, creator: TensorCreator):
        self._action_output = creator.zeros(len(self._actions_descriptor.action_names()), device=creator.device)
        return ActionMonitor(creator, self._actions_descriptor)

    def _step(self):
        self._unit.step(self.inputs.action_in.tensor, self._override_checked, self._actions_values)

    def get_properties(self) -> List[ObserverPropertiesItem]:

        def override_checked(value):
            self._override_checked = value
            return value

        def action_checked(i, value):
            self._actions_values[i] = value
            return value

        return [ObserverPropertiesItem("Override action", 'checkbox', self._override_checked, override_checked)] + [
            self._prop_builder.checkbox(name, self._actions_values[i], partial(action_checked, i))
            for i, name in enumerate(self._actions_descriptor.action_names())
        ]
