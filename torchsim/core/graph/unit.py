import torch
from typing import Union, List

from abc import abstractmethod

from torchsim.core.memory.on_device import OnDevice
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.persistable import Persistable
from torchsim.core.persistence.saver import Saver


class Unit(OnDevice, Persistable):
    _unit_folder_name = 'unit'

    @abstractmethod
    def step(self, *args, **kwargs):
        """Calculates the forward operation of the node."""
        pass

    # TODO (S/L): Add unit property saving/loading mechanism.

    def save(self, parent_saver: Saver):
        saver = parent_saver.create_child(self._unit_folder_name)

        self._save(saver)

        # We need this to be created now so that we can store the tensors inside.
        folder_path = saver.get_full_folder_path()

        saver.description['tensors'] = self.save_tensors(folder_path)

    def load(self, parent_loader: Loader):
        loader = parent_loader.load_child(self._unit_folder_name)

        folder_path = loader.get_full_folder_path()

        self.load_tensors(folder_path, loader.description['tensors'])

        self._load(loader)

    def _save(self, saver: Saver):
        pass

    def _load(self, loader: Loader):
        pass


class InvertibleUnit(Unit):
    @abstractmethod
    def inverse_projection(self, *args, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculates the inverse projection.

        Args:
            In the implementing methods, specify what arguments you need and properly call this
            from node._inverse_projection.

        Returns:
            Either a single tensor, or a list of tensors, depending on what makes sense.
        """
        pass
