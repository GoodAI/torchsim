import torch


class LatentWorld:
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def to_tensor(instances):
        instances = [instance.to_tensor() for instance in instances]
        return torch.stack(instances)
