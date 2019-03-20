import os
from typing import Dict, Any


class Persistor:
    _description_file_name = 'description.yaml'
    _folder: str
    _description: Dict[str, Any]
    _parent: 'Persistor'

    @property
    def description(self) -> Dict[str, Any]:
        return self._description

    def __init__(self, folder: str, parent: 'Persistor'):
        """Initialize the persistor.

        Args:
            folder: The folder name relative to the parent. If this is the root, it should be the absolute path.
            parent: The parent of this persistor. If it's None, this is the root persistor.
        """
        self._parent = parent
        self._folder = folder
        self._description = {}

    def get_full_folder_path(self) -> str:
        """Gets the absolute path of this persistor."""
        if self._parent is None:
            return self._folder

        return os.path.join(self._parent.get_full_folder_path(), self._folder)

    def _get_description_path(self):
        return os.path.join(self.get_full_folder_path(), self._description_file_name)
