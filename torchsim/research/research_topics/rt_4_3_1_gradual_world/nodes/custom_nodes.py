from typing import List, Tuple

import torch

from torchsim.core.nodes import LambdaNode
from torchsim.core.utils.tensor_utils import id_to_one_hot


def create_discount_node(gamma: float, num_predictors: int) -> LambdaNode:
    """Returns a node that implements discount by gamma parameter"""

    def discount(inputs: List[torch.Tensor], outputs: List[torch.Tensor], memory: List[torch.Tensor]):
        cum_val = (1. - gamma) * inputs[0].squeeze() + gamma * memory[0]
        memory[0].copy_(cum_val)
        outputs[0].copy_(cum_val)

    discount_node = LambdaNode(discount, 1, [(num_predictors,)],
                               memory_shapes=[(num_predictors,)],
                               name="Discount")
    return discount_node


def create_arg_min_node(num_predictors: int) -> LambdaNode:
    # argmin
    def argmin(inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
        outputs[0].copy_(id_to_one_hot(inputs[0].argmin(), num_predictors))

    argmin_lambda_node = LambdaNode(argmin, 1, [(num_predictors,)], name="Argmin")
    return argmin_lambda_node


def create_dot_product_node(input_size: int, output_sizes: List[Tuple], name: str) -> LambdaNode:
    # dot
    def dot_product(inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
        outputs[0].copy_(inputs[0].squeeze().dot(inputs[1].squeeze()))

    dot_node = LambdaNode(dot_product, input_size, output_sizes, name=name)
    return dot_node


def create_delay_node(num_predictors: int) -> LambdaNode:
    # delay the data 1 step
    def delay(inputs: List[torch.Tensor], outputs: List[torch.Tensor], memory: List[torch.Tensor]):
        outputs[0].copy_(memory[0])
        memory[0].copy_(inputs[0])

    delay_node = LambdaNode(delay,
                            1,
                            [(num_predictors,)],
                            memory_shapes=[(num_predictors, )],
                            name="Gate Output Delay")
    return delay_node


def create_predictions_gather_node(predicted_indexes: List[int],
                                   num_objects: int = 1,
                                   name: str = "Gather predicted data") -> LambdaNode:
    """Gathers just the parts of the output requested by the indexes (ids of the positions in the vector)"""

    def gather_pred(inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
        latent_input = inputs[0]

        predicted_ids = torch.tensor(predicted_indexes, dtype=torch.long).view(1, -1)
        predicted_expanded = predicted_ids.expand(latent_input.shape[0], -1)

        # possible improvement
        # gathered = latent_input.index_select(1, predicted_indexes)
        gathered = latent_input.gather(1, predicted_expanded)
        # outputs[0].copy_(gathered)
        # TODO only one object supported for now
        outputs[0].copy_(gathered.view(outputs[0].shape))

    gather_predictions = LambdaNode(gather_pred,
                                    n_inputs=1,
                                    # output_shapes=[(num_objects, len(predicted_indexes))],
                                    # TODO only one object supported for now
                                    output_shapes=[(len(predicted_indexes),)],
                                    name=name)
    return gather_predictions

