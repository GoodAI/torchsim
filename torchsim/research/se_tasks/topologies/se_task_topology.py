from abc import ABC

from torchsim.core.graph import Topology
from torchsim.core.nodes.internals.learning_switchable import LearningSwitchable, TestingSwitcher


class TestableTopology(Topology, ABC):
    _is_testing: bool = False

    def __init__(self):
        Topology.__init__(self, device='cuda')

    def is_in_testing_phase(self) -> bool:
        """Finds first TestableSwitch or is_learning method on node and pass the value returned by is_learning.

        Returns:
            True if in testing phase.
        """
        for switch_node in (x for x in self.nodes
                            if isinstance(x, TestingSwitcher) or hasattr(x, 'is_learning')):
            return not switch_node.is_learning()

    def before_step(self):
        if self.is_in_testing_phase() and not self._is_testing:
            # switch to testing -- switch learning off
            self.switch_learning(learning_on=False)
            self._is_testing = True
        elif not self.is_in_testing_phase() and self._is_testing:
            # back to learning
            self.switch_learning(learning_on=True)
            self._is_testing = False

    def switch_learning(self, learning_on: bool):
        for switchable_node in (x for x in self.nodes
                                if isinstance(x, LearningSwitchable) or hasattr(x, 'switch_learning')):
            switchable_node.switch_learning(learning_on)

        self._is_testing = not learning_on
