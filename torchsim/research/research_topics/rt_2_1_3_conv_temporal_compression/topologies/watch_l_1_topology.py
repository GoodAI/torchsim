from argparse import ArgumentParser

from eval_utils import run_just_model
from torchsim.research.research_topics.rt_2_1_3_conv_temporal_compression.topologies.l1_topology import L1Topology

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--persisting-observer-system", default=False, action='store_true')
    args = parser.parse_args()
    t = L1Topology()
    run_just_model(model=t, gui=True, persisting_observer_system=args.persisting_observer_system)
