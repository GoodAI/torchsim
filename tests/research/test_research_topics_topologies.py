import gc
import logging
import os
from importlib import import_module
from os.path import dirname
from typing import List

import pytest

from torchsim.core.graph import Topology
from torchsim.core.nodes import GridWorldNode
from torchsim.core.nodes.internals.grid_world import GridWorldParams
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_base_topology import Task0BaseTopology
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_conv_wide_topology import \
    Task0ConvWideTopology
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_narrow_topology import Task0NarrowTopology
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.node_groups.ta_multilayer_node_group import \
    Nc1r1ClassificationGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.topologies.classification_accuracy_modular_topology import \
    ClassificationAccuracyModularTopology
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.node_groups.ta_multilayer_classification_group import TaMultilayerClassificationGroup
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.topologies.task0_ta_analysis_topology import \
    Task0TaAnalysisTopology
from torchsim.research.research_topics.rt_4_2_1_actions.topologies.goal_directed_template_topology import \
    GoalDirectedTemplateTopology
from tests.testing_utils import discover_child_classes

logger = logging.getLogger(__name__)

skip_topologies = ['SeTaLrfT0', 'GoalDirectedTemplateTopology']


def walk_topics():
    research_module = import_module('torchsim.research.research_topics')
    for root, dirs, _ in os.walk(dirname(research_module.__file__)):
        for topic in dirs:
            if os.path.isdir(os.path.join(root, topic, 'topologies')):
                yield topic


def discover_local_topology_classes(skip_classes: List = None):
    graph_classes = set()
    for topic in walk_topics():
        graph_classes = set.union(
            graph_classes, discover_child_classes(f'torchsim.research.research_topics.{topic}.topologies', Topology,
                                                  skip_classes))

    # TODO: Not testing because they need SpaceEngineers - they are now tested separately in test_se_tasks_topologies
    # graph_classes = set.union(graph_classes, discover_topology_classes(f'torchsim.research.se_tasks.topologies'))

    return [x for x in graph_classes if 'torchsim.research' in x.__module__]


# Put factories here if your topology needs parameters to be constructed.
topology_factories = {
    ClassificationAccuracyModularTopology: lambda: ClassificationAccuracyModularTopology(
        SeNodeGroup(), Nc1r1ClassificationGroup(MultipleLayersParams())),
    Task0TaAnalysisTopology: lambda: Task0TaAnalysisTopology(
        SeNodeGroup(), TaMultilayerClassificationGroup(MultipleLayersParams())),
    GoalDirectedTemplateTopology: lambda: None
}


def instantiate_graph(topology_class):
    """Instantiate the topology.

    Use the factory from topology_factories, or try instantiating the topology without any parameters.
    """
    factory = topology_factories.get(topology_class, topology_class)
    return factory()


@pytest.mark.parametrize('topology_class', discover_local_topology_classes())
def test_topologies_can_be_initialized(topology_class):
    instantiate_graph(topology_class)


@pytest.mark.slow
@pytest.mark.parametrize('topology_class', discover_local_topology_classes(skip_topologies))
def test_topologies_can_run_step(topology_class):
    """Just try that topologies can be initialized and run for one step."""

    instantiate_graph(topology_class).step()


@pytest.mark.slow
def test_task_0_base_topology_step():
    """Specific test for stepping through the Task0BaseTopology."""
    graph = instantiate_graph(Task0BaseTopology)
    graph.nodes[0]._params.save_gpu_memory = True
    graph.step()
    del graph
    gc.collect()

@pytest.mark.slow
def test_task_0_narrow_topology_step():
    """Specific test for stepping through the Task0NarrowTopology."""
    graph = instantiate_graph(Task0NarrowTopology)
    graph.nodes[0]._params.save_gpu_memory = True
    graph.step()
    del graph
    gc.collect()


@pytest.mark.slow
def test_task_0_conv_wide_topology_step():
    """Specific test for stepping through the Task0ConvWideTopology."""
    graph = instantiate_graph(Task0ConvWideTopology)
    graph.nodes[0]._params.save_gpu_memory = True
    graph.step()
    del graph
    gc.collect()
