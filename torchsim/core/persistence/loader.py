import os

from ruamel.yaml import YAML

from torchsim.core.persistence.persistor import Persistor


class Loader(Persistor):
    def __init__(self, folder_name: str, parent: 'Loader' = None):
        super().__init__(folder_name, parent)

        self._load()

    def load_child(self, folder_name: str) -> 'Loader':
        """Initialize a sub-loader.

        Args:
            folder_name: The folder name of the child loader relative to this loader's folder.
        """
        return Loader(folder_name, self)

    def _load(self):
        self._load_description()

    def _load_description(self):
        yaml = YAML()
        description_path = self._get_description_path()
        if os.path.exists(description_path):
            with open(description_path, 'r') as f:
                self._description = yaml.load(f)
