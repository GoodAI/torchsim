from typing import Optional

import torch
from abc import ABC, abstractmethod

from torchsim.core.graph.node_base import TInputs, TOutputs, TInternals
from torchsim.core.graph.slots import InputSlot
from torchsim.core.graph.worker_node_base import InvertibleWorkerNodeWithInternalsBase


class HierarchicalObservableNode(InvertibleWorkerNodeWithInternalsBase[TInputs, TInternals, TOutputs], ABC):
    """An InvertibleWorkerNodeWithInternalsBase which can be observable with the hierarchical observer.

    TODO (Refactor): This should be just an interface, not a base class.
    """

    @property
    @abstractmethod
    def projected_values(self) -> Optional[torch.Tensor]:
        pass

    @property
    @abstractmethod
    def projection_input(self) -> InputSlot:
        pass
