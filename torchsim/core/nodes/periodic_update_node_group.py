from typing import List

from torchsim.core.graph.node_base import TInputs, TOutputs, TInternals
from torchsim.core.graph.node_group import NodeGroupWithInternalsBase
from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime
from torchsim.gui.validators import validate_positive_int


class PeriodicUpdateNodeGroup(NodeGroupWithInternalsBase[TInputs, TInternals, TOutputs]):
    """Node group that runs every n steps.

    Sometimes we want nodes to execute not every step but every n steps. For example, we may want the input data
    to change every five steps to allow multiple iterations of processing of a single frame. The PeriodicUpdateNodeGroup
    provides this functionality, executing with the specified frequency.
    """
    def __init__(self, name: str, update_period: int = 1, inputs: TInputs = None, internals: TInternals = None,
                 outputs: TOutputs = None):
        super().__init__(name, inputs, internals, outputs)
        self._update_period = update_period
        self._step_count = 0

    def step(self):
        if self._step_count % self._update_period == 0:
            super().step()
        self._step_count += 1

    @property
    def update_period(self) -> int:
        return self._update_period

    @update_period.setter
    def update_period(self, value: int):
        validate_positive_int(value)
        self._update_period = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('Update period', type(self).update_period, edit_strategy=disable_on_runtime)
        ]
