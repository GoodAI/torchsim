from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.utils.sequence_generator import SequenceGenerator


class Sequence(Unit):
    """A minimal unit for testing seqs.

    Args:
        creator (TensorCreator): A TensorCreator to initialize the unit.
        seq (SequenceGenerator): A generator for the sequence itself.
    """
    _seq: SequenceGenerator

    def __init__(self, creator: TensorCreator, seq: SequenceGenerator):
        super().__init__(creator.device)
        self._seq = seq
        self._creator = creator

        self.output = creator.zeros(1, dtype=self._float_dtype, device=self._device)
        self.sequence_number = creator.zeros(1, dtype=self._float_dtype, device=self._device)

    def step(self):
        self.output[0] = next(self._seq)
        self.sequence_number[0] = self._seq.current_sequence()


class SequenceOutputs(MemoryBlocks):
    """MemoryBlock Outputs for the SequenceNode.

    Args:
        owner (SequenceNode): The SequenceNode to which these outputs belong.
    """
    def __init__(self, owner: 'SequenceNode'):
        super().__init__(owner)
        self.output = self.create("Output")
        self.sequence_num = self.create("Sequence number")

    def prepare_slots(self, unit: Sequence):
        self.output.tensor = unit.output
        self.sequence_num.tensor = unit.sequence_number


class SequenceNode(WorkerNodeBase[EmptyInputs, SequenceOutputs]):
    """A minimal node for sequences.

    Args:
        seq (SequenceGenerator): A generator for the sequence itself.
        name (str, optional): Name of the node, defaults to 'Sequence'.
    """

    outputs: SequenceOutputs
    _unit: Sequence

    def __init__(self, seq: SequenceGenerator, name="Sequence"):
        super().__init__(name=name, outputs=SequenceOutputs(self))
        self._seq = seq

    def _create_unit(self, creator: TensorCreator):

        return Sequence(creator, self._seq)

    def _step(self):
        self._unit.step()
