import logging
import os
from abc import ABC
from typing import TypeVar, Callable, Any, Iterator, List, Type, Generator, Tuple

import torch

from torchsim.core import get_float

T = TypeVar('T')

logger = logging.getLogger(__name__)


def recurse(obj: T, f: Callable[[T], Any], member_getter: Callable[[T], Iterator[T]]):
    """Applies f to obj and recursively to all items retrieved by member_getter.

    This tracks already visited objects so cyclic references are not followed during the recursion.
    """
    parents = []
    _recurse_f(obj, f, member_getter, parents)
    del parents


def _recurse_f(obj: T, f: Callable[[T], Any], member_getter: Callable[[T], Iterator[T]], parents: List[T]):
    f(obj)
    for member in member_getter(obj):
        if member not in parents:
            _recurse_f(member, f, member_getter, parents + [obj])


class OnDevice(ABC):
    def __init__(self, device):
        self._device = device
        self._on_device_members = []
        self._float_dtype = get_float(self._device)

    def _get_tensors(self):
        return self._get_members_of_type(torch.Tensor)

    def _get_device_members(self):
        for _, member in self._get_members_of_type(OnDevice):
            yield member

    def _get_named_device_members(self) -> Generator[Tuple[str, 'OnDevice'], Any, Any]:
        return self._get_members_of_type(OnDevice)

    def _get_members_of_type(self, member_type: Type):
        for member_name, member in self.__dict__.items():
            if issubclass(type(member), member_type):
                yield member_name, member

    def to_(self, device: str):
        """Shifts torch.Tensor and OnDevice members to either the cpu or gpu.

        Args:
            device (str): Either 'cpu' or 'cuda'
        """
        recurse(self,
                f=lambda on_device: on_device._tensors_to(device),
                member_getter=OnDevice._get_device_members)

    def _tensors_to(self, device: str):
        new_type = get_float(device)
        for tensor_name, tensor in self._get_tensors():
            self.__dict__[tensor_name] = tensor.type(new_type).to(device)

        self._device = device
        self._float_dtype = new_type

    def pin(self):
        """Pins the memory used by torch.Tensor and OnDevice members."""
        recurse(self,
                f=lambda on_device: on_device._pin_tensors(),
                member_getter=OnDevice._get_device_members)

    def _pin_tensors(self):
        for tensor_name, tensor in self._get_tensors():
            self.__dict__[tensor_name] = tensor.pin_memory()

    def copy_to(self, destination: 'OnDevice'):
        """Copies all torch.Tensor and OnDevice members to destination.

        Args:
            destination (OnDevice): Another OnDevice instance to copy the tensors to
        """

        recurse((self, destination),
                f=lambda objects: objects[0]._copy_tensors_to(objects[1]),
                member_getter=lambda objects: zip(*map(OnDevice._get_device_members, objects)))

    def _copy_tensors_to(self, destination: 'OnDevice'):
        assert type(destination) == type(self)

        for tensor_name, tensor in self._get_tensors():
            destination.__dict__[tensor_name].copy_(tensor, non_blocking=True)

    def save_tensors(self, folder):
        """Saves all tensors from this OnDevice into the folder.

        Returns:
            A dictionary describing the structure and file names.
        """
        return self._save_tensors(folder, TensorSaver(), [])

    def _save_tensors(self, folder, tensor_saver, parents):
        description = {}
        parents.append(self)

        for tensor_name, tensor in self._get_tensors():
            file_name = tensor_saver.save(tensor, folder)
            description[tensor_name] = file_name

        for member_name, on_device in self._get_named_device_members():
            if on_device not in parents:
                description[member_name] = on_device._save_tensors(folder, tensor_saver, parents)

        return description

    def load_tensors(self, folder, description):
        for field_name, data in description.items():
            member = self.__dict__[field_name]
            if issubclass(type(member), OnDevice):
                # The member is a OnDevice, let it deserialize itself.
                try:
                    member.load_tensors(folder, data)
                except Exception as e:
                    logger.warning(f"Loading of {field_name} failed with error message: " + e.__str__())

            elif issubclass(type(member), torch.Tensor):
                # The member is a tensor.
                try:
                    tensor = torch.load(os.path.join(folder, data))
                    member.copy_(tensor)
                except Exception as e:
                    logger.warning(f"Loading of tensor {field_name} failed with error message: " + e.__str__())


class TensorSaver:
    """Creates unique names for tensors and dumps them."""
    def __init__(self):
        self._tensor_id = 0

    def save(self, tensor, folder):
        self._tensor_id += 1
        dims = 'x'.join(map(str, tensor.shape))
        file_name = f'{self._tensor_id}_{dims}.pt'
        torch.save(tensor, os.path.join(folder, file_name))

        return file_name
