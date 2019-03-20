from abc import abstractmethod

import numpy as np

from torchsim.core.actions import AgentActionsDescriptor
from torchsim.gui.observer_system import TextObservable


class ActionsDescriptorProvider:
    @abstractmethod
    def get_actions_descriptor(self) -> AgentActionsDescriptor:
        raise NotImplementedError


class ActionsObservable(TextObservable):
    actions_descriptor: AgentActionsDescriptor
    _data: str = ''

    def __init__(self, node: ActionsDescriptorProvider):
        self.actions_descriptor = node.get_actions_descriptor()

    def set_data(self, data: np.array):
        self._data = data

    def get_data(self) -> str:
        action_values = self.actions_descriptor.parse_actions(self._data)
        return "<br>".join(
            (f'{name}: {value}' for name, value in zip(self.actions_descriptor.action_names(), action_values)))
