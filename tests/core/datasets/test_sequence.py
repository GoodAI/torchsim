from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.utils.sequence_generator import SequenceGenerator
from torchsim.core.nodes.sequence_node import SequenceNode
import numpy as np


# Test a range sequence
def test_range_seq():
    seq = [1, 2, 3]
    iterator = SequenceGenerator.from_list(range(1, 4))
    for element in seq * 2:
        assert element == next(iterator)


# Test a deterministic sequence
def test_seq():
    seq = [1, 2, 3, 4, 5, 10, 2, 2]
    iterator = SequenceGenerator.from_list(seq)
    for element in seq * 2:
        assert element == next(iterator)


# Test a sequence node
def test_seq_node():
    seq = [1, 2, 3, 4, 5, 10, 2, 2]
    iterator = SequenceGenerator.from_list(seq)
    node = SequenceNode(seq=iterator)
    node._prepare_unit(AllocatingCreator(device='cpu'))
    for element in seq * 2:
        node.step()
        assert element == node.outputs.output.tensor.item()


# Test multiple seqs with deterministic transitions
def test_multiple():
    seq_1 = [1, 2, 3]
    seq_2 = [9, 8]
    transitions = np.matrix([[0., 1.], [1., 0.]])
    combined_seq = seq_1 + seq_2
    iterator = SequenceGenerator.from_multiple([seq_1, seq_2], transitions)
    for expected, actual in zip(combined_seq, iterator):
        assert expected == actual

