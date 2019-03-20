from copy import deepcopy
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict, Any

from torchsim.core.models.expert_params import ExpertParams, ParamsBase, SpatialPoolerParams, SamplingMethod, \
    TemporalPoolerParams
from torchsim.gui.validators import validate_predicate
from torchsim.significant_nodes import ConvLayer
from torchsim.significant_nodes.conv_layer import TaLayer


@dataclass
class MultipleLayersParams(ParamsBase):
    """ Parameters for N layers.

    Use either list of length N defining the param value for each layer,
    or one value which will be common for all N layers.

    num_conv_layers has to be set to the correct value.
    """

    # by this you specify how many conv_layers the topology should have
    num_conv_layers: int = 1

    # types of convolutional layers
    n_flocks: Union[List[int], int] = 1
    conv_classes: Union[List[TaLayer], TaLayer] = ConvLayer

    # RFs
    rf_size: Union[List[Tuple[int, ...]], Tuple[int, ...]] = (8, 8)
    rf_stride: Union[List[Tuple[int, ...]], Tuple[int, ...]] = (4, 4)

    # SP
    n_cluster_centers: Union[List[int], int] = ExpertParams.n_cluster_centers
    learning_rate: Union[List[float], float] = SpatialPoolerParams.learning_rate
    sp_batch_size: Union[List[int], int] = SpatialPoolerParams.batch_size
    sp_buffer_size: Union[List[int], int] = SpatialPoolerParams.buffer_size
    cluster_boost_threshold: Union[List[int], int] = SpatialPoolerParams.cluster_boost_threshold
    max_boost_time: Union[List[int], int] = SpatialPoolerParams.max_boost_time
    sampling_method: Union[List[SamplingMethod], SamplingMethod] = SamplingMethod.BALANCED
    compute_reconstruction: Union[List[int], int] = ExpertParams.compute_reconstruction

    # TP
    seq_length: Union[List[int], int] = TemporalPoolerParams.seq_length
    max_encountered_seqs: Union[List[int], int] = TemporalPoolerParams.max_encountered_seqs
    max_frequent_seqs: Union[List[int], int] = TemporalPoolerParams.n_frequent_seqs
    seq_lookahead: Union[List[int], int] = TemporalPoolerParams.seq_lookahead
    opp: Union[List[float], float] = TemporalPoolerParams.output_projection_persistence
    exploration_probability: Union[List[float], float] = TemporalPoolerParams.exploration_probability

    def validate_params_for_n_layers(self):

        # iterate through the vars, each of them should contain one value or list of len num_layers
        v = vars(self)
        for var_str in v:
            if isinstance(v[var_str], list):
                validate_predicate(lambda: len(v[var_str]) == self.num_conv_layers,
                                   f'{var_str} should have either one value or list of length {self.num_conv_layers}')

    def _get_param_val(self, param_name: str):
        """Get value of a parameter with a given name (either list of values or a single value)"""
        v = vars(self)
        if param_name not in v:
            raise ValueError(f'{param_name} not found in MultipleLayerParams')
        return v[param_name]

    def read_param(self, param_name: str, layer_no: int):
        """Read parameter value of a given name for a given layer"""
        self.validate_params_for_n_layers()

        param_val = self._get_param_val(param_name)

        if isinstance(param_val, list):
            # list of values?
            if layer_no >= len(param_val):
                raise ValueError(f'layer_no: {layer_no} out of range of parameters list, range is {len(param_val)}')
            return param_val[layer_no]

        # common value for all layers
        return param_val

    def read_list_of_params(self, param_name: str) -> Union[List[Any], Tuple[Any]]:
        """Read list of parameters of a given name for all layers"""
        self.validate_params_for_n_layers()

        param_val = self._get_param_val(param_name)

        if isinstance(param_val, list):
            return param_val.copy()  # copy, so that others cannot modify the internal data through the pointer

        return [param_val] * self.num_conv_layers

    def convert_to_expert_params(self):
        """Parse from the MultipleLayerParams to list of ExpertParams"""
        self.validate_params_for_n_layers()

        params_list = []

        for layer_id in range(self.num_conv_layers):
            params = ExpertParams()
            params.flock_size = 1

            # spatial
            params.n_cluster_centers = self.read_param('n_cluster_centers', layer_id)

            params.spatial.buffer_size = self.read_param('sp_buffer_size', layer_id)
            params.spatial.batch_size = self.read_param('sp_batch_size', layer_id)

            params.spatial.cluster_boost_threshold = self.read_param('cluster_boost_threshold', layer_id)
            params.spatial.max_boost_time = self.read_param('max_boost_time', layer_id)

            params.spatial.learning_rate = self.read_param('learning_rate', layer_id)
            params.spatial.sampling_method = self.read_param('sampling_method', layer_id)

            params.compute_reconstruction = self.read_param('compute_reconstruction', layer_id)

            # temporal
            params.temporal.seq_length = self.read_param('seq_length', layer_id)
            params.temporal.seq_lookahead = self.read_param('seq_lookahead', layer_id)
            params.temporal.max_encountered_seqs = self.read_param('max_encountered_seqs', layer_id)
            params.temporal.n_frequent_seqs = self.read_param('max_frequent_seqs', layer_id)
            params.temporal.exploration_probability = self.read_param('exploration_probability', layer_id)

            # done
            params_list.append(params)

        return params_list

    def change(self, **kwargs):
        """Create new instance with the same values as this instance, change just some selected params.

        Args:
            **kwargs: change arbitrary parameter values here

        Returns: a new class instance with all the values the same, but those specified in the params of this method
        are changed
        """
        # clone to the new instance
        result = deepcopy(self)
        
        if kwargs is not None:
            for key, value in kwargs.items():
                if key not in vars(result):
                    raise ValueError(f'key {key} not found in vars(MultipleLayersParams); a typo?')
                vars(result)[key] = value
        return result

    def get_params_as_short_names(self, prefix: str = "") -> Dict[str, Any]:
        """Return all params as a dict {prefix+name:value}"""
        params = {}

        v = vars(self)
        for param_name in v:
            # add value to the dictionary
            # TODO might define short names of parameters here (for the legend) if necessary
            params[prefix + param_name] = v[param_name]
        return params



