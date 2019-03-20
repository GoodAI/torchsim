import logging

from eval_utils import parse_test_args, create_observer_system, run_experiment_with_ui

from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.experiment_runner import UiExperimentRunner
from torchsim.research.research_topics.rt_3_2_1_symbolic_input_words.templates.symbolic_input_template import \
    SymbolicInputTemplate
from torchsim.research.research_topics.rt_3_2_1_symbolic_input_words.topologies.symbolic_input_words_topology import \
    SymbolicInputWordsTopology

logger = logging.getLogger(__name__)


def run_measurement(name, topology_parameters, args, debug: bool = False):
    """"Runs the experiment with specified params, see the parse_test_args method for arguments"""

    # exp_params = Params

    scaffolding = TopologyScaffoldingFactory(SymbolicInputWordsTopology)
    template = SymbolicInputTemplate("Symbolic input test")
    experiment_params = ExperimentParams(max_steps=0)
    experiment = Experiment(template, scaffolding, topology_parameters, experiment_params)

    run_experiment_with_ui(experiment)


def run_words(args):
    name = "Words"
    params = [
        {}
    ]
    run_measurement(name, params, args)


if __name__ == '__main__':
    arg = parse_test_args()
    run_words(arg)
