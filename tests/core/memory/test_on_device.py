import tempfile

import pytest
import torch

from torchsim.core.memory.on_device import OnDevice, recurse
from torchsim.core.utils.tensor_utils import same


class StubBase:
    """This exists just to test the mixin's behavior from the multiple inheritance point of view."""

    def __init__(self, foo):
        self.foo = foo


class OnDeviceStub(StubBase, OnDevice):
    def __init__(self, device, device_member=None):
        StubBase.__init__(self, 42)
        OnDevice.__init__(self, device)
        self.tensor = torch.tensor([1, 2, 3], dtype=torch.float64, device=device)
        self.device_member = device_member
        if device_member is not None:
            # Check that circular references are handled correctly.
            device_member.device_member = self


def test_recurse():
    objects = []
    on_device = create_on_device()
    recurse(on_device, f=lambda x: objects.append(x), member_getter=lambda x: x._get_device_members())

    for obj in [on_device, on_device.device_member]:
        assert obj in objects
        objects.remove(obj)

    assert 0 == len(objects)


def create_on_device():
    return OnDeviceStub('cpu', device_member=OnDeviceStub('cpu'))


def move_to_and_check(on_device, device):
    on_device.to_(device)

    assert on_device._device == device
    assert on_device.tensor.device.type == device
    assert on_device.device_member._device == device
    assert on_device.device_member.tensor.device.type == device


def test_to_():
    on_device = create_on_device()

    move_to_and_check(on_device, 'cuda')
    move_to_and_check(on_device, 'cpu')


def test_pin():
    on_device = create_on_device()

    on_device.pin()

    assert on_device.tensor.is_pinned()
    assert on_device.device_member.tensor.is_pinned()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_copy_(device):
    on_device = create_on_device()
    on_device2 = create_on_device()

    on_device.to_(device)
    on_device2.to_(device)

    new_tensor = torch.rand_like(on_device.tensor)

    on_device.tensor = new_tensor
    on_device.device_member.tensor = new_tensor

    on_device.copy_to(on_device2)

    assert same(new_tensor, on_device2.tensor)
    assert same(new_tensor, on_device2.device_member.tensor)


def test_failing_copy_():
    on_device = create_on_device()
    on_device2 = create_on_device()

    new_tensor = torch.rand_like(on_device.tensor)

    on_device.tensor = new_tensor
    on_device.device_member.tensor = new_tensor

    on_device2.to_('cuda')

    with pytest.raises(RuntimeError):
        on_device.copy_to(on_device2)
        assert same(on_device.tensor, on_device2.tensor)


def test_save_load():
    on_device = create_on_device()

    # Different tensors.
    on_device2 = create_on_device()
    on_device2.tensor.random_()
    on_device2.device_member.tensor.random_()

    with tempfile.TemporaryDirectory() as folder:
        description = on_device.save_tensors(folder)
        on_device2.load_tensors(folder, description)

    assert same(on_device.tensor, on_device2.tensor)
    assert same(on_device.device_member.tensor, on_device2.device_member.tensor)
