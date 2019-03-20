from argparse import ArgumentParser

from eval_utils import run_just_model
from torchsim.research.research_topics.rt_2_1_3_conv_temporal_compression.topologies.l3_conv_topology import \
    L3ConvTopology, L3SpConvTopology

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sp-only", default=False, action='store_true')
    parser.add_argument("--persisting-observer-system", default=False, action='store_true')
    args = parser.parse_args()

    if args.sp_only:
        t = L3SpConvTopology()
    else:
        t = L3ConvTopology()
    run_just_model(model=t, gui=True, persisting_observer_system=args.persisting_observer_system)
