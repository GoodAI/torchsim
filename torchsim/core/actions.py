from abc import abstractmethod
from typing import List
import numpy as np


class AgentActionsDescriptor:
    ACTION_COUNT: int

    @abstractmethod
    def action_names(self) -> List[str]:
        pass

    @abstractmethod
    def parse_actions(self, actions: np.array) -> List[bool]:
        pass


class SpaceEngineersActionsDescriptor(AgentActionsDescriptor):

    def __init__(self):
        self._action_names = [
            'UP',
            'DOWN',
            'FORWARD',
            'BACKWARD',
            'LEFT',
            'RIGHT',
            'JUMP',
            'CROUCH',
            'USE',
            'TURN_LIGHTS',
        ]
        SpaceEngineersActionsDescriptor.ACTION_COUNT = len(self._action_names)

    def action_names(self) -> List[str]:
        return self._action_names

    def parse_actions(self, actions: np.array) -> List[bool]:
        return [val > 0 for val in actions]
