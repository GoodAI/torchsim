import logging

from eval_utils import run_just_model
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.node_groups.dummy_model_group import DummyModelGroup
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.topologies.task0_ta_analysis_topology import \
    Task0TaAnalysisTopology

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    num_conv_layers = 1
    use_top_layer = True

    cf_easy = [1, 2, 3, 4]

    params = [
        {'se_group': {'class_filter': cf_easy},
         'model': {}},
        {'se_group': {'class_filter': cf_easy},
         'model': {}
         }
    ]

    scaffolding = TopologyScaffoldingFactory(Task0TaAnalysisTopology, se_group=SeNodeGroup, model=DummyModelGroup)

    run_just_model(scaffolding.create_topology(**params[0]), gui=True, persisting_observer_system=True)

