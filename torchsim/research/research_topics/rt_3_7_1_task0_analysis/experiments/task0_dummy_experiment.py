import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass

from eval_utils import parse_test_args, run_experiment, run_experiment_with_ui
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_controller import TrainTestComponentParams
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.nodes import DatasetSeObjectsNode
from torchsim.research.experiment_templates2.task0_ta_analysis_template import Task0TaAnalysisTemplate, Task0TaAnalysisParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.node_groups.dummy_model_group import DummyModelGroup
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.topologies.task0_ta_analysis_topology import \
    Task0TaAnalysisTopology

logger = logging.getLogger(__name__)


@dataclass
class Params:
    experiment_params: Task0TaAnalysisParams
    train_test_params: TrainTestComponentParams


debug_params = Params(Task0TaAnalysisParams(measurement_period=1,
                                            num_classes=DatasetSeObjectsNode.label_size(),
                                            num_layers=2,  # has to be overwritten by actual num layers later
                                            sp_evaluation_period=2,
                                            show_conv_agreements=True),
                      TrainTestComponentParams(num_testing_phases=3,
                                               num_testing_steps=160,
                                               overall_training_steps=30))

middle_params = Params(Task0TaAnalysisParams(measurement_period=1,
                                             num_classes=DatasetSeObjectsNode.label_size(),
                                             num_layers=2,
                                             sp_evaluation_period=2),
                       TrainTestComponentParams(num_testing_phases=6,
                                                num_testing_steps=200,
                                                overall_training_steps=1500))

full_params = Params(Task0TaAnalysisParams(measurement_period=1,
                                           num_classes=DatasetSeObjectsNode.label_size(),
                                           num_layers=2,
                                           sp_evaluation_period=100),
                     TrainTestComponentParams(num_testing_phases=25,
                                              num_testing_steps=500,
                                              overall_training_steps=25000))


def run_measurement(name, topology_parameters, args, exp_pars):
    """"Runs the experiment with specified params, see the parse_test_args method for arguments"""

    scaffolding = TopologyScaffoldingFactory(Task0TaAnalysisTopology,
                                             se_group=SeNodeGroup,
                                             model=DummyModelGroup)

    template = Task0TaAnalysisTemplate("Task 0 layer-wise stats and classification accuracy",
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


def run_debug_comparison(args, num_conv_layers: int, exp_params):
    name = "Learning-rate-debug"

    cf_easy = [1, 2, 3, 4]
    size = SeDatasetSize.SIZE_24

    params = [
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {}
        },
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {}
        }
    ]

    exp_params.experiment_params.num_layers = num_conv_layers + 1  # needs to be there

    run_measurement(name, params, args, exp_params)


if __name__ == '__main__':
    arg = parse_test_args()

    run_debug_comparison(arg, 1, debug_params)
