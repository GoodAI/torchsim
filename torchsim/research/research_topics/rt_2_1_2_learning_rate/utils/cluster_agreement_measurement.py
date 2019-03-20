from typing import List, Union

import torch

from torchsim.core.eval.metrics.cluster_agreement import cluster_agreement
from torchsim.utils.template_utils.template_helpers import list_int_to_long_tensors


class ClusterAgreementMeasurement:

    NO_VALUE: float = -0.1  # indicates that there is no cluster agreement output available yet

    def __init__(self):
        pass

    @staticmethod
    def compute_cluster_agreements(test_outputs: Union[List[torch.Tensor], List[List[int]]],
                                   compute_self_agreement: bool = False) -> List[List[float]]:
        """
        List (for each testing phase) of lists of tensors (one tensor containing output IDs from the run)
        Args:
            compute_self_agreement: if true, the cluster agreement with itself will be computed as well
            test_outputs: outputs during testing, either List of Lists of scalars or List of torch.LongTensors
        Returns: List (for each clustering~in a phase) of lists of floats (agreement with all other clusterings~phases)
        """

        if type(test_outputs[0]) is list:
            test_outputs = list_int_to_long_tensors(test_outputs)

        assert len(test_outputs) > 1, 'test_outputs have to be at least from two testing phases'

        # for each clustering, we have series of values of cluster agreements with others
        cluster_id_series = []

        for my_id, my_clustering in enumerate(test_outputs):
            # for each clustering, we will measure how much my clustering agrees with it
            # ..except that the clustering was made before me (cluster agreement with itself included)
            my_agreements = []
            for other_id, other_clustering in enumerate(test_outputs):
                if my_id < other_id:
                    agreement = cluster_agreement(my_clustering, other_clustering)
                    my_agreements.append(agreement)
                elif my_id == other_id and compute_self_agreement:
                    agreement = cluster_agreement(my_clustering, other_clustering)
                    my_agreements.append(agreement)
                else:
                    my_agreements.append(ClusterAgreementMeasurement.NO_VALUE)

            cluster_id_series.append(my_agreements)

        return cluster_id_series

