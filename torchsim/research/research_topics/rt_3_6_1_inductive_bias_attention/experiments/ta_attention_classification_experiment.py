import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from eval_utils import parse_test_args, run_experiment, run_experiment_with_ui
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_controller import TrainTestComponentParams
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.nodes import DatasetSeObjectsNode
from torchsim.research.experiment_templates2.task0_train_test_classification_acc_template import \
    Task0TrainTestClassificationAccTemplate, Task0TrainTestClassificationAccParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.topologies.classification_accuracy_modular_topology import \
    ClassificationAccuracyModularTopology
from torchsim.research.research_topics.rt_3_6_1_inductive_bias_attention.node_groups.attention_classification_group import \
    AttentionClassificationGroup

logger = logging.getLogger(__name__)


@dataclass
class Params(ParamsBase):
    experiment_params: Task0TrainTestClassificationAccParams
    train_test_params: TrainTestComponentParams


debug_params = Params(Task0TrainTestClassificationAccParams(measurement_period=1,
                                                            num_classes=DatasetSeObjectsNode.label_size(),
                                                            sp_evaluation_period=2),
                      TrainTestComponentParams(num_testing_phases=6,
                                               num_testing_steps=12,
                                               overall_training_steps=360))

full_params = Params(Task0TrainTestClassificationAccParams(measurement_period=1,
                                                           num_classes=DatasetSeObjectsNode.label_size(),
                                                           sp_evaluation_period=100),
                     TrainTestComponentParams(num_testing_phases=10,
                                              num_testing_steps=1500,
                                              overall_training_steps=25000))


def run_measurement(name, topology_parameters, args, debug: bool = False):
    """"Runs the experiment with specified params, see the parse_test_args method for arguments"""

    exp_pars = debug_params if debug else full_params

    scaffolding = TopologyScaffoldingFactory(ClassificationAccuracyModularTopology,
                                             se_group=SeNodeGroup,
                                             model=AttentionClassificationGroup)

    template = Task0TrainTestClassificationAccTemplate("Task 0 classification accuracy with and without attention",
                                                       exp_pars.experiment_params,
                                                       exp_pars.train_test_params)

    runner_parameters = ExperimentParams(max_steps=exp_pars.train_test_params.max_steps,
                                         save_cache=args.save,
                                         load_cache=args.load,
                                         clear_cache=args.clear,
                                         calculate_statistics=not args.computation_only,
                                         experiment_folder=args.alternative_results_folder)

    experiment = Experiment(template, scaffolding, topology_parameters, runner_parameters)

    logger.info(f'Running model: {name}')
    if args.run_gui:
        run_experiment_with_ui(experiment)
    else:
        run_experiment(experiment)

    if args.show_plots:
        plt.show()


def run(args, use_single_layer: bool = True, use_two_layers: bool = True, debug: bool = False):
    name = "bottom_up_attention"

    image_size = (24, 24, 3)
    input_data_size = int(np.prod(image_size))
    # n_cluster_centers = 20
    n_middle_layer_cluster_centers = 200
    n_cluster_centers = 200
    # top_layer_params = MultipleLayersParams(n_cluster_centers=40)

    # cf_easy = [1, 2, 3, 4]

    assert use_single_layer or use_two_layers, "At least one of use_single_layer and user_two_layers has to be true"

    params = []

    if use_single_layer:
        params += [
            {'se_group': {'class_filter': None},
             'model': {'use_attention': True,
                       'num_labels': 20,
                       'input_data_size': input_data_size,
                       # 'top_layer_params': top_layer_params,
                       'n_cluster_centers': n_cluster_centers}},
            {'se_group': {'class_filter': None},
             'model': {'use_attention': False,
                       'num_labels': 20,
                       'input_data_size': input_data_size,
                       # 'top_layer_params': top_layer_params,
                       'n_cluster_centers': n_cluster_centers}}
        ]

    if use_two_layers:
        params += [
            {'se_group': {'class_filter': None},
             'model': {'use_attention': True,
                       'num_labels': 20,
                       'input_data_size': input_data_size,
                       # 'top_layer_params': top_layer_params,
                       'n_cluster_centers': n_cluster_centers,
                       'use_middle_layer': True,
                       'n_middle_layer_cluster_centers': n_middle_layer_cluster_centers,
                       'use_temporal_pooler': True}},
            {'se_group': {'class_filter': None},
             'model': {'use_attention': True,
                       'num_labels': 20,
                       'input_data_size': input_data_size,
                       # 'top_layer_params': top_layer_params,
                       'n_cluster_centers': n_cluster_centers,
                       'use_middle_layer': True,
                       'n_middle_layer_cluster_centers': n_middle_layer_cluster_centers,
                       'use_temporal_pooler': False}},
            {'se_group': {'class_filter': None},
             'model': {'use_attention': False,
                       'num_labels': 20,
                       'input_data_size': input_data_size,
                       # 'top_layer_params': top_layer_params,
                       'n_cluster_centers': n_cluster_centers,
                       'use_middle_layer': True,
                       'n_middle_layer_cluster_centers': n_middle_layer_cluster_centers,
                       'use_temporal_pooler': True}},
            {'se_group': {'class_filter': None},
             'model': {'use_attention': False,
                       'num_labels': 20,
                       'input_data_size': input_data_size,
                       # 'top_layer_params': top_layer_params,
                       'n_cluster_centers': n_cluster_centers,
                       'use_middle_layer': True,
                       'n_middle_layer_cluster_centers': n_middle_layer_cluster_centers,
                       'use_temporal_pooler': False}}
        ]

    run_measurement(name, params, args, debug)


if __name__ == '__main__':
    arg = parse_test_args()
    run(arg, debug=False)
