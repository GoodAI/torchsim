import pytest
from pytest import raises

from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.dataset_alphabet_node import DatasetAlphabetParams, DatasetAlphabetSequenceProbsModeParams
from torchsim.core.nodes.multi_dataset_alphabet_node import MultiDatasetAlphabetUnit, MultiDatasetAlphabetNode


class TestMultiDatasetAlphabetUnit:

    n_worlds = 3

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_create_node(self, device):
        unit = MultiDatasetAlphabetUnit(AllocatingCreator(device), DatasetAlphabetParams(
            symbols="abcd",
            sequence_probs=DatasetAlphabetSequenceProbsModeParams(seqs=['abcd'])
        ), n_worlds=self.n_worlds)
        assert [self.n_worlds, 7, 5] == list(unit.output_data.shape)


class TestMultiDatasetAlphabetNode:
    @pytest.mark.parametrize('params, should_pass', [
        (DatasetAlphabetParams(
            symbols="abcd",
            sequence_probs=DatasetAlphabetSequenceProbsModeParams(seqs=['abc'])
        ), True),
        (DatasetAlphabetParams(
            symbols="abcd",
            sequence_probs=DatasetAlphabetSequenceProbsModeParams(seqs=['abc', 'ae'])
        ), False)
    ])
    def test_validate_params_sequence_probs_validate_throws(self, params, should_pass):
        node = MultiDatasetAlphabetNode(params)
        if should_pass:
            node.validate()
        else:
            with raises(NodeValidationException):
                node.validate()
