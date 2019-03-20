from typing import List

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.core.graph import Topology
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.significant_nodes import ConvLayer, SpConvLayer


def make_test_params():

    cp = MultipleLayersParams()
    cp.compute_reconstruction = True
    cp.conv_classes = ConvLayer
    cp.sp_buffer_size = 6000
    cp.sp_batch_size = 4000
    cp.learning_rate = 0.02
    cp.cluster_boost_threshold = 1000
    cp.max_encountered_seqs = 10000
    cp.max_frequent_seqs = 1000
    cp.seq_length = 4
    cp.seq_lookahead = 1
    cp.num_conv_layers = 1

    cp.n_cluster_centers = 400
    cp.rf_size = (8, 8)
    cp.rf_stride = (8, 8)

    tp = MultipleLayersParams()
    tp.n_cluster_centers = 250
    tp.sp_buffer_size = 4000
    tp.sp_batch_size = 500
    tp.learning_rate = 0.2
    tp.cluster_boost_threshold = 1000
    tp.compute_reconstruction = True

    common_params = {
        'conv_layers_params': cp,
        'top_layer_params': tp,
        'image_size': SeDatasetSize.SIZE_64,
        'class_filter': [1, 2, 3],
        'noise_amp': 0.0,
        'model_seed': None,
        'baseline_seed': None
    }

    changing_params = [
        {'conv_layers_params': cp.change(learning_rate=0.1)},
        {'conv_layers_params': cp.change(learning_rate=0.2), 'top_layer_params': tp.change(learning_rate=0.1)},
        {'conv_layers_params': cp.change(learning_rate=0.3)}
    ]

    params = ExperimentTemplateBase.add_common_params(changing_params, common_params)

    return params, common_params, changing_params


class MockFluffyTopology(Topology):

    constructor_param: List[int]
    noise_amp: float

    def __init__(self,
                 noise_amp: float = 0.99,
                 constructor_param: List[int] = [3, 2, 1]):
        super().__init__('cpu')
        self.constructor_param = constructor_param
        self.noise_amp = noise_amp


def test_unpack_params():
    """Test the params are correctly unpacked to the dictionary {str: value}, with the data classes removed"""

    params, _, _ = make_test_params()

    unpacked_run_params = [ExperimentTemplateBase._unpack_params(run_pars) for run_pars in params]

    separator = '_'
    conv_prefix = 'conv_layers_params'[0:4]+separator
    top_prefix = 'top_layer_params'[0:4]+separator

    for run_params in unpacked_run_params:
        assert run_params[conv_prefix+'num_conv_layers'] == 1
        assert run_params[top_prefix+'n_cluster_centers'] == 250

    assert unpacked_run_params[0][conv_prefix+'learning_rate'] == 0.1
    assert unpacked_run_params[1][conv_prefix+'learning_rate'] == 0.2
    assert unpacked_run_params[2][conv_prefix+'learning_rate'] == 0.3


def test_extract_constructor_params():
    """We should be able to extract the constructor params of the class and their value

    Parse the params into the str: Any dictionary,
    merge it with the params for each run and then start doing extraction of things that are duplicate
    """

    params, _, _ = make_test_params()

    constructor_params = ExperimentTemplateBase._get_default_topology_params(MockFluffyTopology)

    # test we've extracted all the params and with correct values
    mock_instance = MockFluffyTopology()
    assert len(constructor_params) == 2
    assert constructor_params['constructor_param'] == mock_instance.constructor_param
    assert constructor_params['noise_amp'] == mock_instance.noise_amp

    complete_params = ExperimentTemplateBase.add_common_params(params, constructor_params)

    assert len(params) == len(complete_params)
    assert len(params[0]) + 1 == len(complete_params[0])  # one non-covered param is in the constructor

    # go thorough the complete params, for each param:
    #   if it is in the manually specified params, the value should be preserved
    #   if it was not specified, the value should be the one extracted from the constructor
    for complete_run_param, run_param in zip(complete_params, params):
        for key, value in complete_run_param.items():
            if key in run_param:
                assert complete_run_param[key] == run_param[key]
            else:
                assert key == 'constructor_param'
                assert value == mock_instance.constructor_param


def test_split_to_unique_and_changing_params():
    """We should be able to split unpacked params to the keys that have changing values and the others"""

    params, common_params, changing_params = make_test_params()

    # does not necessarily have to be here, but just to test everything:
    complete_params = ExperimentTemplateBase.add_common_params(
        params,
        ExperimentTemplateBase._get_default_topology_params(MockFluffyTopology))

    # add this common_param to the corresponding list for testing
    mock_instance = MockFluffyTopology()
    common_params['constructor_param'] = mock_instance.constructor_param

    unpacked_all = [ExperimentTemplateBase._unpack_params(run_par) for run_par in complete_params]
    unpacked_common = ExperimentTemplateBase._unpack_params(common_params)
    unpacked_changing = [ExperimentTemplateBase._unpack_params(run_par) for run_par in changing_params]

    del unpacked_common['conv_learning_rate']  # these things are changing
    del unpacked_common['top__learning_rate']

    # this should filter-out things that are changed between runs (different at least in one run)
    constant_params_filtered = ExperimentTemplateBase._find_constant_parameters(unpacked_all)
    assert constant_params_filtered == unpacked_common

    # this should remove the constant params from the complete list
    changing_params_filtered = ExperimentTemplateBase._remove_params(unpacked_all, constant_params_filtered)

    assert len(changing_params_filtered) == 3
    for cp, complete_par in zip(changing_params_filtered, unpacked_all):
        assert len(cp) == 2
        assert cp['conv_learning_rate'] == complete_par['conv_learning_rate']
        assert cp['top__learning_rate'] == complete_par['top__learning_rate']

    print("done")


def test_params_to_string():
    """We should be able to parse everything into reasonable string"""

    params, _, _ = make_test_params()

    # modify the default params with some more interesting configuration
    num_layers = 3
    conv_classes = [SpConvLayer, ConvLayer, SpConvLayer]

    for run_par in params:
        run_par['conv_layers_params'] = run_par['conv_layers_params'].change(num_conv_layers=num_layers,
                                                                             conv_classes=conv_classes)

    cp = params[0]['conv_layers_params']
    assert cp.conv_classes == conv_classes
    assert cp.num_conv_layers == num_layers

    unpacked = [ExperimentTemplateBase._unpack_params(run_par) for run_par in params]

    conv_layers_str = ExperimentTemplateBase._param_value_to_string(unpacked[0]['conv_conv_classes'])

    assert conv_layers_str == '[SpConvLayer, ConvLayer, SpConvLayer]'

    # complete parsed string (other formating like the one above could be tested here)
    result = ExperimentTemplateBase.parameters_to_string(unpacked)






