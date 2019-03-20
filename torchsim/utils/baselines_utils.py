import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler

from typing import Dict, Any, Callable, Iterable, List, Set
from math import floor

from functools import partial
from torchsim.core import get_float
from torchsim.core.utils.tensor_utils import same
from torchsim.gui.observables import ObserverPropertiesItem


def output_size(h_w: Any, kernel_size: Any = 1, stride: int = 1,
                pad: int = 0, dilation: int = 1) -> tuple:
    """Calculates output size of a convolutional or max pooling layer.

    Args:
        h_w: a value/tuple (h, w) defining input size into the layer
        kernel_size: size of convolutional or pooling kernel
        stride: stride of the window used (default is kernel_size for pooling)
        pad: padding size
        dilation: dilation parameter

    Returns:
        height and width of output
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1)/stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1)/stride) + 1)
    return h, w


class BaselinesParams:
    """ Abstract Params Class for baselines."""

    _default_params: Dict[str, Any] = {}

    def __init__(self, kwargs: Dict[str, Any]) -> None:
        """Constructs an object storing parameters defined in kwargs.

        Args:
            kwargs: A dictionary of parameters to be stored
                    e.g. {'lr': 2.5e-4, 'eps': 1e-5,  'alpha': 0.99}
        """
        self.set_params(kwargs)

    def set_params(self, kwargs: Dict[str, Any]) -> None:
        """Set parameters from input.

        Args:
            kwargs: A dictionary of parameters to be stored
                    e.g. {'lr': 2.5e-4, 'eps': 1e-5,  'alpha': 0.99}
        """
        for name in kwargs:
            setattr(self, name, kwargs[name])

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Returns a dictionary of default parameters.

        Returns:
            dictionary of default parameters
        """
        return cls._default_params


class ObservationStorage:
    """Universal Storage for Baselines.

    A class that allows for the storage and retrieval of arbitrary data over time
    over a predefined fixed length sequence period. Depiction:

                                            + First     +  Last
                                            v           v
                                          +-+-----------++
                 Observation Storage  +-> | 0| 1| 2| 3| 4|  Round 1
                                          +--------------+
                     Buffer size = 5        +-----------+
                                            |
        No. of observations seen = 9      +-v------------+
                                          | 4| 5| 6| 7| 8|  Round 2
                                          +--------------+

    Sampling of data from storage is performed via arbitrary sampling strategy
    chosen at runtime. All sampling methods implemented in PyTorch and through
    its interface are supported.
    """
    _step: int
    _buffer_size: int

    def __init__(self, buffer_size: int, obs_types: Dict[str, Any]) -> None:
        """Constructs a storage object.

        Args:
            buffer_size: size of the buffer storage e.g. 128
            obs_types: a dictionary of observation objects to store, together
                       with corresponding sizes/shapes. First value in shape/size
                       must equal buffer_size.
                       e.g. {'x': (128, 3, 28, 28), 'y': (128, 2)}
        """
        self._buffer_size = buffer_size
        self._obs_types = obs_types
        self._step = 0
        
        for name in obs_types:
            setattr(self, name, torch.zeros(*obs_types[name]))

    def to(self, device: str, kwvars: Dict[str, Any] = None) -> None:
        """Transfer storage to a device (CPU/GPU).

        Args:
            device: string indicating which device to transfer to
            kwvars: a dictionary of attributes this function is applied to
        """
        if kwvars is None:
            attributes = self._obs_types
        else:
            attributes = kwvars

        for name in attributes:
            setattr(self, name, getattr(self, name).to(device).type(get_float(device)))

    def insert(self, data: Dict[str, Any]) -> None:
        """Insert a set of data items into storage.

        Args:
            data: a dictionary with data to store
                  e.g. {'x': torch.rand((128, 3, 64, 64)), 'y': torch.rand((128,2))}
        """
        assert len(data) == len(self._obs_types), 'Number of items to store != storage size!'
        for name in data:
            getattr(self, name)[self._step].copy_(data[name])
        self._step = (self._step + 1) % self._buffer_size

    def after_update(self, attributes: Set[str]) -> None:
        """Action performed after every model update.

        Copies the last item in storage to the front, ready for next round of data.

        Args:
            attributes: dictionary of objects this should be performed on
                        e.g. {'x', 'y']
        """
        for name in attributes:
            getattr(self, name)[0].copy_(getattr(self, name)[-1])

    def generator(self, batch_size: int, sampler: Callable = SubsetRandomSampler,
                  **kwargs) -> Iterable:
        """Creates a data generator from current data stored in the storage.

        Generator samples data according to sampler passed in as argument.

        Args:
            batch_size: size of batches the generator produces
            sampler: sampler type determining sampling strategy
            **kwargs: any other parameters to be passed to sampler

        Returns:
            a generator yielding batches of data produced by a sampler
        """
        assert self._obs_types[next(iter(self._obs_types))][0] % batch_size == 0
        _sampler = BatchSampler(sampler(**kwargs), batch_size, drop_last=False)
        batch = []
        # TODO: parameterize to allow for [:-1] for some
        for indices in _sampler:
            for name in self._obs_types:
                batch.append(getattr(self, name).view(
                    -1, *self._obs_types[name][1:])[indices])
            yield batch

    def __eq__(self, other):
        """Storage has normal attributes and tensors, compare both cases properly"""
        if not isinstance(other, ObservationStorage):
            return False

        attrs = vars(self)

        for attribute_name in attrs:
            a = getattr(self, attribute_name)
            b = getattr(other, attribute_name)

            if type(a) is torch.Tensor:
                if not same(a, b):
                    return False
            else:
                if a != b:
                    return False

        return True


def parse_vars(params: BaselinesParams) -> List[ObserverPropertiesItem]:
    """Parses attributes of an object and places them into a
    list of ObserverPropertiesItems suitable for GUI.

    Useful for automatically adding params to OUI GUI

    Args:
        params: dictionary of parameters to add to GUI

    Returns:
        list of ObserverPropertiesItem used by the GUI

    """
    def _setter(varname: str, value: Any) -> Any:
        var = type(getattr(params, varname))
        setattr(params, varname, var.__call__(value))
        return value

    varlist: List[ObserverPropertiesItem] = list()
    for varname in vars(params):
        vartype = 'number'
        if type(getattr(params, varname)) is str:
            vartype = 'text'
        elif type(getattr(params, varname)) is bool:
            vartype = 'checkbox'
        varlist.append(
            ObserverPropertiesItem(varname, vartype,
                                   getattr(params, varname),
                                   partial(_setter, varname))
        )
    return varlist
