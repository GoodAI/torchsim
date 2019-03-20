import pytest

from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket
from torchsim.core.utils.tensor_utils import same
from torchsim.research.research_topics.rt_3_1_lr_subfields.random_subfield_node import RandomSubfieldForkNode
from tests.core.nodes.node_unit_test_base import NodeTestBase


@pytest.mark.flaky(reruns=2)
class TestRandomSubfieldForkNode(NodeTestBase):
    def _generate_input_tensors(self):
        yield [
            self._creator.full((2, 2, 3, 3, 1), fill_value=2, device=self._device, dtype=self._dtype),
        ]

    def _generate_expected_results(self):
        yield [self._creator.full((2, 2, 3, 3, 1), fill_value=2, dtype=self._dtype, device=self._device)] * 5

    def _create_node(self):
        return RandomSubfieldForkNode(n_outputs=5, n_samples=7, first_non_expanded_dim=2)

    @staticmethod
    def _same(expected, result) -> bool:
        return expected.shape == result.shape and \
               same(result[0], result[1])

    def _run_node_for_steps(self, node: RandomSubfieldForkNode, sources, check_results: bool = True):
        for step, (inputs, expected_results) in enumerate(
                zip(self._generate_input_tensors(), self._generate_expected_results())):
            self._replace_data_in_inputs(sources, inputs)
            node.step()
            if check_results:
                results = self._extract_results(node)
                self._check_results(expected_results, results, step)

                # test 1. result is different from 2.

                assert not same(results[0], results[1])

                # test inverse projection for 1. output

                input_packet = InversePassOutputPacket(results[0], node.outputs[0])
                output_packet = node.recursive_inverse_projection_from_output(input_packet)

                inverse_tensor = output_packet[0].tensor
                assert (inverse_tensor == 0).any() == 1
                assert (inverse_tensor > 0).any() == 1
                assert (inverse_tensor > 2).any() == 0
