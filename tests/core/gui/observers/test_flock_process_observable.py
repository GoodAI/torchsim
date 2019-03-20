import torch

from torchsim.core import FLOAT_TYPE_CPU
from torchsim.core.models.flock import Process
from torchsim.core.utils.tensor_utils import same
from torchsim.gui.observers.flock_process_observable import FlockProcessObservable


class ProcessStub(Process):
    def run(self):
        pass

    def _check_dims(self, *args):
        pass


def test_process_observer_init():
    process_initialized = False
    tensor = None

    indices = torch.tensor([0])

    def process_provider():
        if process_initialized:
            return ProcessStub(indices, do_subflocking=False)
        return None

    def tensor_provider(process):
        return tensor

    observer = FlockProcessObservable(flock_size=2, process_provider=process_provider, tensor_provider=tensor_provider)

    tensor = observer.get_tensor()
    assert tensor is None

    process_initialized = True
    tensor = observer.get_tensor()
    assert tensor is None

    # The first expert runs and provides data, the second does not run so the observer shows NaN.
    tensor = torch.tensor([0], dtype=FLOAT_TYPE_CPU)
    tensor = observer.get_tensor()
    assert same(tensor, torch.tensor([0, float('nan')], dtype=FLOAT_TYPE_CPU))

    # The second expert runs and provides data, the first does not run so the observer shows NaN.
    indices = torch.tensor([1])
    tensor = torch.tensor([0], dtype=FLOAT_TYPE_CPU)
    tensor = observer.get_tensor()
    assert same(tensor, torch.tensor([float('nan'), 0], dtype=FLOAT_TYPE_CPU))

    # NaNs, NaNs everywhere.
    process_initialized = False
    tensor = observer.get_tensor()
    assert same(tensor, torch.tensor([float('nan'), float('nan')], dtype=FLOAT_TYPE_CPU))
