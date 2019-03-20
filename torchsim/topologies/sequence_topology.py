import numpy as np

from torchsim.core.nodes import SequenceNode
from torchsim.core.graph import Topology
from torchsim.core.utils.sequence_generator import SequenceGenerator


class SequenceTopology(Topology):
    """Super simple model for testing seqs.

    A minimal model for testing seqs.
    """
    def __init__(self):
        super().__init__(device='cpu')
        generator = SequenceGenerator([[1, 2, 3], [4, 5, 6]], np.array([0.5, 0.5]))
        seq_node = SequenceNode(generator)

        self.add_node(seq_node)
