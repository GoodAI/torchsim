from abc import ABC

from torchsim.core.models.flock.process import Process


class TPProcess(Process, ABC):
    """This is here only for code organization purposes (to match the structure of SPProcess-es)."""
    pass
