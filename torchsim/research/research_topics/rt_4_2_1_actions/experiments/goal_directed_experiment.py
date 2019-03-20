import logging

from eval_utils import parse_test_args, run_experiment_with_ui, run_experiment
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.nodes.internals.grid_world import GridWorldParams, ResetStrategy
from torchsim.research.research_topics.rt_4_2_1_actions.node_groups.two_experts_group import TwoExpertsGroup
from torchsim.research.research_topics.rt_4_2_1_actions.templates.goal_directed_template import GoalDirectedTemplate
from torchsim.research.research_topics.rt_4_2_1_actions.topologies.goal_directed_template_topology import \
    GoalDirectedTemplateTopology, GoalDirectedTemplateTopologyParams

logger = logging.getLogger(__name__)


def run_measurement(topology_parameters, args, run_debug: bool, avg_reward_window_size: int = 100,
                    run_gui: bool = True):
    max_steps = 100 if run_debug else 40000

    scaffolding = TopologyScaffoldingFactory(GoalDirectedTemplateTopology,
                                             model=TwoExpertsGroup,
                                             params=GoalDirectedTemplateTopologyParams)

    template = GoalDirectedTemplate("Goal Directed Behavior - Comparision of TA Hierarchies",
                                    avg_reward_window_size=avg_reward_window_size)

    runner_parameters = ExperimentParams(max_steps=max_steps,
                                         save_cache=args.save,
                                         load_cache=args.load,
                                         clear_cache=args.clear,
                                         calculate_statistics=not args.computation_only,
                                         experiment_folder=args.alternative_results_folder)

    experiment = Experiment(template, scaffolding, topology_parameters, runner_parameters)

    if run_gui:
        run_experiment_with_ui(experiment)
    else:
        run_experiment(experiment)


def three_rooms_tiny(run_debug, run_gui, n_parallel_runs):
    world_params = GridWorldParams(map_name='MapThreeRoomTiny', reset_strategy=ResetStrategy.ANYWHERE)
    # 4 actions per each world state.
    # c_n_ccs = world_params.get_n_unique_visible_egocentric_states() * 4
    c_n_ccs = 44
    params = [
        {
            'model': {'c_n_ccs': c_n_ccs, 'c_seq_length': 5, 'c_seq_lookahead': 3, 'c_buffer_size': 10000,
                      'p_seq_length': 7, 'p_seq_lookahead': 5, 'p_n_ccs': 5, 'flock_size': n_parallel_runs},
            'params': {'use_egocentric': False, 'n_parallel_runs': n_parallel_runs, 'world_params': world_params},
        }
    ]
    run_measurement(params, parse_test_args(), run_debug, avg_reward_window_size=499, run_gui=run_gui)


def reward_hint(run_debug, run_gui, n_parallel_runs):
    world_params = GridWorldParams(map_name='Friston', reward_switching=True)

    c_n_ccs = 19
    params = [
        {
            'model': {'c_n_ccs': c_n_ccs, 'c_seq_length': 4, 'c_seq_lookahead': 2, 'c_buffer_size': 7000,
                      'p_seq_length': 6, 'p_seq_lookahead': 4, 'p_n_ccs': 6, 'flock_size': n_parallel_runs},
            'params': {'use_egocentric': True, 'n_parallel_runs': n_parallel_runs, 'world_params': world_params},
        }
    ]
    run_measurement(params, parse_test_args(), run_debug, avg_reward_window_size=499, run_gui=run_gui)


if __name__ == '__main__':
    debug = False
    gui = True
    # three_rooms_tiny(debug, gui, n_parallel_runs=5)
    reward_hint(debug, gui, n_parallel_runs=5)
