from typing import List

from ruamel.yaml import YAML

from torchsim.core.persistence.persistor import Persistor
from torchsim.utils.os_utils import create_dir


class Saver(Persistor):
    _children: List['Saver']

    def __init__(self, folder_name: str, parent: 'Saver' = None):
        """See Persistor.__init__.

        Note that this also creates the folder so that any custom data can be stored in there.
        """
        super().__init__(folder_name, parent)

        self._children = []
        self._create_folder()

    def create_child(self, folder_name: str) -> 'Saver':
        """Initialize a sub-saver.

        Args:
            folder_name: The folder name of the child saver relative to this saver's folder.
        """
        child = Saver(folder_name, self)
        self._children.append(child)
        return child

    def save(self):
        """Saves the saver and its children on the disk."""
        for child in self._children:
            child.save()

        self._save_description()

    def _create_folder(self):
        full_path = self.get_full_folder_path()
        create_dir(full_path)

    def _save_description(self):
        if len(self._description) > 0:
            yaml = YAML()
            with open(self._get_description_path(), 'w') as f:
                yaml.dump(self._description, f)
