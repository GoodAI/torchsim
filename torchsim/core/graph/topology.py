import logging

from torchsim.core import get_float
from torchsim.core.graph.id_generator import IdGenerator
from torchsim.core.graph.node_group import GenericNodeGroup
from torchsim.core.memory.tensor_creator import MeasuringCreator, AllocatingCreator
from torchsim.core.model import Model
from torchsim.utils.seed_utils import set_global_seeds


logger = logging.getLogger(__name__)


class MemoryBlockSizesNotConvergingException(Exception):
    pass


class Topology(GenericNodeGroup, Model):
    _seed: int
    _topology_folder_name = 'graph'

    # This was never parametrized, we can keep it here.
    _max_block_update_iterations = 3  # set to 3 - it's used just in context_size setup and it should not iterate.
    # In the second step it's value is set and then one iteration is needed to check stationary value.
    # If you need it, let's have a design meeting on this feature.

    _measuring_creator: MeasuringCreator
    _allocating_creator: AllocatingCreator

    device: str

    def __init__(self, device: str):
        super().__init__('Graph', 0, 0)
        self.device = device
        self.float_dtype = get_float(self.device)

        self._measuring_creator = MeasuringCreator()
        self._allocating_creator = AllocatingCreator(device)

        self._seed = None
        self._do_before_step = None
        self._visible_nodes = set()

        self._id_generator = IdGenerator()

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value

    def _update_memory_blocks(self):
        # Set seeds to provide defaults for nodes which didn't care to set the seeds themselves.
        set_global_seeds(self._seed)

        for _ in range(self._max_block_update_iterations):
            self.allocate_memory_blocks(self._measuring_creator)
            changed = self.detect_dims_change()
            if not changed:
                break
        else:
            # If the cycle didn't break, we can't run the model.
            raise MemoryBlockSizesNotConvergingException()

        # All memory block now contain the surrogates with the dimensions that don't change anymore,
        # so these can now be used for the real allocation.
        self.allocate_memory_blocks(self._allocating_creator)

        # Reset global seeds before simulation run.
        set_global_seeds(self._seed)

    def step(self):
        """This overrides NodeBase as the topology also handles some additional logic."""
        if not self.is_initialized():
            self.prepare()

        self.before_step()

        self._step()

        self.after_step()

    def prepare(self):
        # This is only run once. That will change later when we allow for dynamically changing topologies.
        self.assign_ids()
        self.order_nodes()
        self._update_memory_blocks()
        self.validate()

    def assign_ids(self):
        self._assign_ids_to_nodes(self._id_generator)

    def stop(self):
        self.release_memory_blocks()

    def _get_persistence_name(self):
        return self._topology_folder_name

    def before_step(self):
        pass

    def after_step(self):
        pass
