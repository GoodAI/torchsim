import logging

from eval_utils import parse_test_args, run_experiment_with_ui, run_experiment
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.research.research_topics.rt_4_1_1_gradual_learning_basic.templates.gradual_learning_basic_template import \
    GradualLearningBasicTemplate, OneShotLearningExperimentComponent, NotForgettingExperimentComponent, \
    KnowledgeReuseExperimentComponent
from torchsim.research.research_topics.rt_4_1_1_gradual_learning_basic.topologies.gradual_learning_basic_topology import \
    GradualLearningBasicTopology, GradualLearningBasicTopologyParams, GLExperimentParams, NotForgettingExperimentParams

logger = logging.getLogger(__name__)


def run_measurement(name, topology_parameters, args, debug: bool = False):
    """"Runs the experiment with specified params, see the parse_test_args method for arguments"""

    # exp_params = Params
    # def param_pass(gate_input_context_multiplier: int) -> Any:
    #     return {'gate_input_context_multiplier': gate_input_context_multiplier}
    # scaffolding = TopologyScaffoldingFactory(GradualLearningBasicTopology, params=param_pass)

    scaffolding = TopologyScaffoldingFactory(GradualLearningBasicTopology, params=GradualLearningBasicTopologyParams)
    template = GradualLearningBasicTemplate("Gradual learning basic")
    params: NotForgettingExperimentParams = topology_parameters[0]['params']['experiment_params'].params
    max_steps = params.phase_1_steps + params.phase_2_steps + params.phase_3_steps + 1
    experiment_params = ExperimentParams(max_steps=max_steps,
                                         save_cache=args.save,  # add --save param
                                         load_cache=args.load,  # add --load param
                                         )
    experiment = Experiment(template, scaffolding, topology_parameters, experiment_params)

    # run_experiment_with_ui(experiment, auto_start=True)
    run_experiment(experiment)


def run_this_experiment(args):
    flock_size = 1000
    flock_split = 500
    name = "Gradual learning basic"
    params = [
        # # Not forgetting experiment (ex4)
        # {'params': {
        #     'gate_input_context_multiplier': 8,
        #     'flock_size': flock_size,
        #     'flock_split': flock_split,
        #     'experiment_params': GLExperimentParams(
        #         component=NotForgettingExperimentComponent,
        #         params=NotForgettingExperimentParams(
        #             phase_1_steps=10000,
        #             phase_2_steps=7000,
        #             phase_3_steps=5000
        #         )
        #     )}},

        # # One shot learning experiment (ex5)
        # {'params': {
        #     'gate_input_context_multiplier': 8,
        #     'flock_size': flock_size,
        #     'flock_split': flock_split,
        #     'experiment_params': GLExperimentParams(
        #         component=OneShotLearningExperimentComponent,
        #         params=NotForgettingExperimentParams(
        #             phase_1_steps=5000,
        #             phase_2_steps=0,
        #             phase_3_steps=3000
        #         )
        #     )}},

        # Knowledge reuse experiment (ex4)
        {'params': {
            'gate_input_context_multiplier': 8,
            'flock_size': flock_size,
            'flock_split': flock_split,
            'experiment_params': GLExperimentParams(
                component=KnowledgeReuseExperimentComponent,
                params=NotForgettingExperimentParams(
                    phase_1_steps=5000,
                    phase_2_steps=5000,
                    phase_3_steps=5000
                )
            )}},
    ]
    run_measurement(name, params, args)


if __name__ == '__main__':
    arg = parse_test_args()
    run_this_experiment(arg)
