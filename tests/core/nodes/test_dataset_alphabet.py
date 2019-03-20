import pytest
from pytest import raises

from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.dataset_alphabet_node import DatasetAlphabetUnit, DatasetAlphabetParams, \
    DatasetAlphabetSequenceProbsModeParams, DatasetAlphabetNode


class TestDatasetAlphabetUnit:
    @staticmethod
    def label_generator(unit: DatasetAlphabetUnit):
        while True:
            unit.step()
            yield int(unit.output_label[0])

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_create_node(self, device):
        unit = DatasetAlphabetUnit(AllocatingCreator(device), DatasetAlphabetParams(
            symbols="abcd",
            sequence_probs=DatasetAlphabetSequenceProbsModeParams(seqs=['abcd'])
        ))
        assert [4, 7, 5] == list(unit.all_symbols.shape)
        assert [7, 5] == list(unit.output_data.shape)

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_mode_sequence_probs(self, device):
        unit = DatasetAlphabetUnit(AllocatingCreator(device), DatasetAlphabetParams(
            symbols="abcd",
            sequence_probs=DatasetAlphabetSequenceProbsModeParams(
                seqs=['abc']
            )
        ))
        generator = self.label_generator(unit)
        result = [next(generator) for _ in range(7)]
        assert [0, 1, 2, 0, 1, 2, 0] == result

    @pytest.mark.parametrize('symbols, text, expected', [
        ("ab", "bba", [1, 1, 0]),
        ("abcd", "acbbd", [0, 2, 1, 1, 3]),
        ("abcd", "", [])
    ])
    def test_convert_string_to_positions(self, symbols, text, expected):
        result = DatasetAlphabetUnit.convert_string_to_positions(symbols, text)
        assert expected == result

    def test_convert_string_to_positions_throws(self):
        with raises(ValueError):
            DatasetAlphabetUnit.convert_string_to_positions("abc", "d")


class TestDatasetAlphabetNode:
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
        node = DatasetAlphabetNode(params)
        if should_pass:
            node.validate()
        else:
            with raises(NodeValidationException):
                node.validate()
