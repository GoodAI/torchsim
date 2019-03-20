from typing import Optional

import torch
from abc import ABC, abstractmethod

from torchsim.core.memory.on_device import OnDevice
from torchsim.core.models.expert_params import ParamsBase, ExpertParams
from torchsim.core.models.flock.process import Process


class Flock(OnDevice, ABC):

    _params: ExpertParams

    """A base class for a flock."""
    def __init__(self, params, device):
        super().__init__(device)
        self.validate_params(params)

        self.buffer = None
        self._params = params

    def _determine_forward_pass(self, data: torch.Tensor, mask: Optional[torch.Tensor]):
        pass

    def _determine_learning(self, forward_mask):
        pass

    @staticmethod
    def _run_process(process: Process):
        """Runs and integrates a process, doing nothing if None is received."""
        if process is None:
            return

        process.run_and_integrate()

    @abstractmethod
    def validate_params(self, params: ParamsBase):
        """Validate the params. Potentially also with respect to the state of the object.
        (Maybe we will not need it?)."""
