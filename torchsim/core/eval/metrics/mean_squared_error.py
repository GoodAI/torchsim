import torch


def mse_loss(input: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    # noinspection PyUnresolvedReferences
    return torch.nn.functional.mse_loss(input, reconstruction)
