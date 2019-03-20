import torch


def one_hot_entropy(codes: torch.Tensor):
    """Compute the entropy of a representation (code) using one-hot vectors.

    The entropy is computed under the following assumptions:
        1. The code consists of a number of one-hot vectors
        2. The component (one-hot) vectors have different distributions
        3. The component vectors are considered independent

    The entropy is computed separately for each component vector. The total entropy is then computed as the sum of the
    entropies of the component codes.

    In TA experiments, a component vector corresponds to the SP output of an expert and an element in the one-hot
    component vector to a cluster center.

    Args:
        codes: A tensor containing the codes with dimensions (sample, component, element)

    Returns:
        The entropy of the code (sum of component entropies)
    """
    probabilities = codes.sum(0) / codes.shape[0]  # Sum over samples / number of samples
    probabilities[probabilities == 0] = 1  # Handle zero probabilities (log(1) == 0)
    entropy = -(probabilities * probabilities.log2()).sum()  # Sum component entropies
    return entropy
