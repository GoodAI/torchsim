import logging

from eval_utils import run_topology_factory_with_ui, observer_system_context, run_topology_with_ui
from torchsim.core.graph import Topology
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.topologies.se_dataset_sp_representation import \
    SEDatasetSPRepresentationTopology
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_narrow_topology import \
    Task0NarrowTopology
from torchsim.research.research_topics.rt_2_1_3_conv_temporal_compression.topologies.l3_conv_topology import L3ConvTopology
from torchsim.research.research_topics.rt_4_3_1_gradual_world.topologies.gl_nn_world_topology import setup_demo_model
from torchsim.topologies.SampleCollectionTopology import SEDatasetSampleCollectionTopology
from torchsim.topologies.SeDatasetObjectsTopology import SeDatasetObjectsTopology
from torchsim.topologies.bottom_up_attention_topology import BottomUpAttentionTopology
from torchsim.topologies.context_test_topology import ContextTestTopology
from torchsim.topologies.disentangled_world_node_topology import DisentangledWorldNodeTopology
from torchsim.topologies.expert_hierarchy_topology import ExpertHierarchyTopology
from torchsim.topologies.expert_topology import ExpertTopology
from torchsim.topologies.gl_nn_topology import GlNnTopology, GlFakeGateNnTopology
from torchsim.topologies.gl_nn_topology import GlNnTopology
from torchsim.topologies.goal_directed_narrow_hierarchy_topology import GoalDirectedNarrowHierarchyTopology
from torchsim.topologies.goal_directed_topology import GoalDirectedTopology
from torchsim.topologies.grid_world_topology import GridWorldTopology
from torchsim.topologies.looping_topology import LoopingTopology
from torchsim.topologies.lrf_object_detection_topology import LrfObjectDetectionTopology
from torchsim.topologies.mnist_topology import MnistTopology
from torchsim.topologies.mse_demo_topology import MseDemoTopology
from torchsim.topologies.multi_dataset_alphabet_topology import MultiDatasetAlphabetTopology
from torchsim.topologies.network_flock_topology import NetworkFlockTopology
from torchsim.topologies.nnet_topology import NNetTopology
from torchsim.topologies.random_number_topology import RandomNumberTopology
from torchsim.topologies.receptive_field_topology import ReceptiveFieldTopology
from torchsim.topologies.se_toyarch_debug_topology import SeToyArchDebugTopology
from torchsim.topologies.sequence_mnist_topology import SequenceMnistTopology
from torchsim.topologies.sequence_topology import SequenceTopology
from torchsim.topologies.sp_topologies import SpatialPoolerTopology, SpatialPoolerHierarchy, ConvSpatialPoolerTopology, \
    SpatialTemporalPoolerTopology
from torchsim.topologies.bouncing_ball_topology import BouncingBallTopology
from torchsim.topologies.debug_agent_topology import DebugAgentTopology
from torchsim.topologies.noise_topology import RandomNoiseTopology, RandomNoiseOnlyTopology
import argparse

from torchsim.topologies.single_expert_expermental_topology import SingleExpertExperimentalTopology
from torchsim.topologies.switch_topology import SwitchTopology
from torchsim.topologies.ta_actions_grid_world_topology import TaActionsGridWorldTopology
from torchsim.topologies.ta_exploration_grid_world_topology import TaExplorationGridWorldTopology

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="se_dataset_navigation")
    parser.add_argument("--seed", type=int, help="Global random seed.")
    parser.add_argument("--random-seed", action='store_true', help="Use random seed. Overrides the seed argument.")
    args = parser.parse_args()

    # UI persistence storage file - set to None to turn off the persistence
    storage_file = 'observers.yaml'

    if args.random_seed:
        seed = None
    else:
        seed = args.seed if args.seed is not None else 1337

    def topology_factory(key) -> Topology:
        if key == "SP":
            # works
            model = SpatialPoolerTopology()
        elif key == "SPTP":
            # not updated
            model = SpatialTemporalPoolerTopology()
        elif key == "seq":
            # not updated
            model = SequenceTopology()
        elif key == "noise":
            # works
            model = RandomNoiseTopology()
        elif key == "noise_only":
            # works
            model = RandomNoiseOnlyTopology()
        elif key == "random_num":
            # works
            model = RandomNumberTopology()
        elif key == "switch":
            # works
            model = SwitchTopology()
        elif key == "mnist":
            # works
            model = MnistTopology()
        elif key == "mnist_seq":
            # works
            model = SequenceMnistTopology()
        elif key == "SP_hierarchy":
            # works
            model = SpatialPoolerHierarchy()
        elif key == "bouncing_ball":
            # works
            model = BouncingBallTopology()
        elif key == "debug_agent":
            # works (probably)
            model = DebugAgentTopology()
        elif key == "se_ta":
            # WIP
            model = SeToyArchDebugTopology()
        elif key == "expert":
            # works
            model = ExpertTopology()
        elif key == "expert_hierarchy":
            # works
            model = ExpertHierarchyTopology()
        elif key == "se_dataset_navigation":
            model = SEDatasetSPRepresentationTopology()
        elif key == "se_dataset_objects":
            model = SeDatasetObjectsTopology()
        elif key == "grid_world":
            model = GridWorldTopology()
        elif key == "looping":
            model = LoopingTopology()
        elif key == "conv_sp":
            model = ConvSpatialPoolerTopology()
        elif key == "mse":
            model = MseDemoTopology()
        elif key == "actions_grid_world":
            model = TaActionsGridWorldTopology()
        elif key == "rtx_narrow":
            model = Task0NarrowTopology()
        elif key == "sample_collection":
            model = SEDatasetSampleCollectionTopology()
        elif key == "nnet":
            model = NNetTopology()
        elif key == "l3_conv_topology":
            model = L3ConvTopology()
        elif key == "receptive_field":
            model = ReceptiveFieldTopology()
        elif key == "context_test":
            model = ContextTestTopology()
        elif key == "exploration_grid_world":
            model = TaExplorationGridWorldTopology()
        elif key == "bottom_up_attention":
            model = BottomUpAttentionTopology()
        elif key == "lrf_object_detection_topology":
            model = LrfObjectDetectionTopology()
        elif key == "disentangled_world_node_topology":
            model = DisentangledWorldNodeTopology()
        elif key == "net":
            model = NetworkFlockTopology()
        elif key == "goal_directed":
            model = GoalDirectedTopology()
        elif key == "goal_directed_narrow":
            model = GoalDirectedNarrowHierarchyTopology()
        elif key == "gl_nn":
            model = GlNnTopology()
        elif key == "gl_nn_fg":
            model = GlFakeGateNnTopology()
        elif key == "super_hard":
            model = SingleExpertExperimentalTopology()
        elif key == "multi_dataset_alphabet":
            model = MultiDatasetAlphabetTopology()
        elif key == "nn-gradual-learning-demo":
            model = setup_demo_model()
        else:
            raise NotImplementedError

        model.assign_ids()

        return model

    # Create simulation, it is registers itself to observer_system
    with observer_system_context(storage_file) as observer_system:
        # run_topology_factory_with_ui(topology_factory=topology_factory,
        #                              topology_params={'key': args.model},
        #                              seed=seed,
        #                              max_steps=0,
        #                              auto_start=False,
        #                              observer_system=observer_system)
        # A simpler version which doesn't recreate the topology, just restarts.
        run_topology_with_ui(topology=topology_factory(args.model),
                             seed=seed,
                             max_steps=0,
                             auto_start=False,
                             observer_system=observer_system)

    print('Running simulation, press enter to quit.')
    input()


if __name__ == '__main__':
    main()
