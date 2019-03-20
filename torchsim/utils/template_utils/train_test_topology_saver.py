import logging
import os

from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.persistable import Persistable
from torchsim.core.persistence.saver import Saver

logger = logging.getLogger(__name__)


class PersistableSaver:
    """Saves and loads the persistable for purpose of train/test splitting ExperimentTemplate"""

    _saver: Saver
    _loader: Loader

    def __init__(self, adapter_name: str):
        persistence_path = self._get_persistence_location(adapter_name)

        self._saver = Saver(persistence_path)

        if not os.path.exists(persistence_path):
            logger.error(f"There is no saved model at location {persistence_path}")

        self._loader = Loader(persistence_path)

    def save_data_of(self, persistable: Persistable):
        """
        Saves a switchable to a default location
        Args:
            persistable: the switchable to be saved
        """
        persistable.save(self._saver)
        self._saver.save()
        logger.info('Persistable saved')

    def load_data_into(self, persistable: Persistable):
        """
        Loads a persistable from a default location
        Args:
            persistable: the persistable into which the data will be loaded
        """
        try:
            persistable.load(self._loader)
        except FileNotFoundError:
            logger.exception(f"Loading of persistable failed")

        logger.info('Persistable loaded')

    @staticmethod
    def _get_persistence_location(adapter_name: str):
        return os.path.join(os.getcwd(), 'data', 'stored', adapter_name)
