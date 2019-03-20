from torchsim.core.graph import Topology
from torchsim.core.nodes.dataset_alphabet_node import DatasetAlphabetParams, DatasetAlphabetSequenceProbsModeParams
from torchsim.core.nodes.multi_dataset_alphabet_node import MultiDatasetAlphabetNode


class MultiDatasetAlphabetTopology(Topology):

    def __init__(self):
        super().__init__(device = 'cuda')

        dataset_params = DatasetAlphabetParams(symbols="abcd123456789", padding_right=1,
                                               sequence_probs=DatasetAlphabetSequenceProbsModeParams(
                                                   seqs=['abc', '123', '456'],
                                               ))

        dataset = MultiDatasetAlphabetNode(dataset_params, n_worlds=4)

        self.add_node(dataset)



