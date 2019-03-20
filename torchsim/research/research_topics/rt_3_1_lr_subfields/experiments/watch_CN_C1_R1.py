from argparse import ArgumentParser

from eval_utils import run_just_model
from torchsim.core.graph import Topology
from torchsim.research.research_topics.rt_3_1_lr_subfields.node_groups import CN_C1_R1
from torchsim.research.research_topics.rt_3_1_lr_subfields.node_groups.CN_C1_R1 import SCN_SC1_R1
from torchsim.significant_nodes import BallEnvironment, BallEnvironmentParams, SeEnvironmentParams, SEEnvironment

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--persisting-observer-system", default=False, action='store_true')
    parser.add_argument("--use-sp-only", default=False, action='store_true')
    parser.add_argument("--space-engineers", default=False, action='store_true')
    args = parser.parse_args()

    if args.use_sp_only:
        topology_class = SCN_SC1_R1
    else:
        topology_class = CN_C1_R1

    params = {
        "name": "C5C1R1",
        "l_0_cluster_centers": 10,
        "l_1_cluster_centers": 20,
        "l_0_rf_dims": (3, 3),
        "l_0_rf_stride": None,
        "l_1_rf_dims": (2, 2),
        "sp_n_cluster_centers": 10
    }

    t = Topology('cuda')
    if args.space_engineers:
        se_params = SeEnvironmentParams(shapes=list(range(SeEnvironmentParams.n_shapes)))
        params["bottom_layer_size"] = 5
        params["env_size"] = se_params.env_size
        params["label_length"] = se_params.n_shapes
        env = SEEnvironment(se_params)
    else:
        params["bottom_layer_size"] = 2
        params["env_size"] = (24, 24)
        params["label_length"] = 3
        env = BallEnvironment(BallEnvironmentParams())

    cnc1r1 = topology_class(**params)
    t.add_node(cnc1r1)
    t.add_node(env)
    env.outputs.connect_automatically(cnc1r1.inputs)

    run_just_model(model=t, gui=True, persisting_observer_system=args.persisting_observer_system)
