import pytest

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.significant_nodes import ConvLayer
from torchsim.topologies.toyarch_groups.ncmr1_group import NCMR1Group


def test_validate_params_for_n_layers():
    """Create default params, set custom values expecting 3-layer network, check validation"""

    params = MultipleLayersParams()
    params.seq_length = [1, 2, 3]
    params.n_cluster_centers = [1, 2, 3]
    params.cluster_boost_threshold = 123
    params.num_conv_layers = 3

    params.validate_params_for_n_layers()

    with pytest.raises(AttributeError):
        params.nonexisting_field = 23


def test_create_flock_params():
    """Read from the params class and convert to list of params class instances"""

    params = MultipleLayersParams()
    params.seq_length = [1, 2, 3]
    params.n_cluster_centers = [1, 2, 3]
    params.cluster_boost_threshold = 123
    params.num_conv_layers = 3

    params_list = params.convert_to_expert_params()
    assert len(params_list) == 3

    conv_classes = params.read_list_of_params('conv_classes')
    assert len(conv_classes) == 3

    conv_classes += [params.read_param('conv_classes', 0)]
    assert len(conv_classes) == 4

    conv_classes += params.read_list_of_params('conv_classes')
    assert len(conv_classes) == 7

    num_cc = params.read_list_of_params('n_cluster_centers')
    assert len(num_cc) == 3
    n_cc = params.read_param('n_cluster_centers', 1)
    assert n_cc == 2

    cbt = params.read_list_of_params('cluster_boost_threshold')
    assert isinstance(cbt, list)
    assert len(cbt) == 3

    top_params = MultipleLayersParams()
    cbt = top_params.read_list_of_params('cluster_boost_threshold')
    assert isinstance(cbt, list)
    assert len(cbt) == 1


def test_default_top_params():
    """Default params should be set for 1 layer (i.e. the top-level one)"""

    params = MultipleLayersParams()
    params_list = params.convert_to_expert_params()

    assert len(params_list) == 1


def test_group_constructor():

    conv_layers_params = MultipleLayersParams()
    conv_layers_params.n_cluster_centers = [20]

    top_layer_params = MultipleLayersParams()

    group = NCMR1Group(conv_layers_params, top_layer_params)

    assert group._num_layers == 2


def test_empty_change():

    a = MultipleLayersParams()
    a.learning_rate = 0.345
    a.n_cluster_centers = [1, 2, 3]

    b = a.change()
    assert b.learning_rate == a.learning_rate
    assert a.n_cluster_centers == b.n_cluster_centers

    b.learning_rate = 0.1
    a.n_cluster_centers = [1, 2]

    assert b.learning_rate != a.learning_rate
    assert a.n_cluster_centers != b.n_cluster_centers


def test_add_common_params():
    """Test the helper function that adds default parameters to all experiment runs"""

    cp = MultipleLayersParams()
    cp.compute_reconstruction = True
    cp.conv_classes = [ConvLayer, ConvLayer]
    cp.num_conv_layers = 2

    tp = MultipleLayersParams()
    tp.n_cluster_centers = 250

    # class filter changed
    params = [
        {'class_filter': [1, 2]},
        {'class_filter': [1, 2, 3]},
        {'class_filter': [1]},
        {}
    ]

    common_params = {
        'conv_layers_params': cp,
        'top_layer_params': tp,
        'image_size': SeDatasetSize.SIZE_64,
        'noise_amp': 0.12,
        'model_seed': None,
        'baseline_seed': None,
        'class_filter': [9]
    }

    p = ExperimentTemplateBase.add_common_params(params, common_params)

    assert len(p) == 4
    assert len(p[0]) == 7
    assert p[0]['class_filter'] == [1, 2]
    assert p[1]['class_filter'] == [1, 2, 3]
    assert p[2]['class_filter'] == [1]
    assert p[3]['class_filter'] == [9]

    for param in p:
        assert param['noise_amp'] == 0.12


def test_change_params():
    default_params = MultipleLayersParams()
    default_params.learning_rate = 0.997
    default_params.n_cluster_centers = [2, 3]
    default_params.seq_length = 5

    sequence = [
        {'conv_layers_params': default_params.change(n_cluster_centers=[1, 2])},
        {'conv_layers_params': default_params.change(n_cluster_centers=[1, 2, 3], learning_rate=0.11)}
    ]

    assert default_params.n_cluster_centers == [2, 3]
    assert sequence[0]['conv_layers_params'].n_cluster_centers == [1, 2]
    assert sequence[1]['conv_layers_params'].n_cluster_centers == [1, 2, 3]

    assert sequence[0]['conv_layers_params'].learning_rate == 0.997
    assert sequence[1]['conv_layers_params'].learning_rate == 0.11







