import argparse
import logging

from eval_utils import observer_system_context, run_topology_with_ui
from torchsim.core.graph import Topology
from torchsim.core.utils.sequence_generator import SequenceGenerator, diagonal_transition_matrix
from torchsim.topologies.gl_nn_topology import GlNnTopology, GlFakeGateNnTopology
import numpy as np

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gl_nn")
    parser.add_argument("--sequences", default="s0", nargs='*')
    parser.add_argument("--seed", type=int, help="Global random seed.")
    parser.add_argument("--n-predictors", type=int, default=2, help="Number of predictors.")
    parser.add_argument("--random-seed", action='store_true', help="Use random seed. Overrides the seed argument.")
    args = parser.parse_args()

    # UI persistence storage file - set to None to turn off the persistence
    storage_file = 'observers.yaml'

    if args.random_seed:
        seed = None
    else:
        seed = args.seed if args.seed is not None else 1337

    def topology_factory(key, seq_list) -> Topology:
        fourth_sequence_excluded = np.array(
            [
                [0.8, 0.1, 0.1, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.1, 0.1, 0.8, 0.0],
                [1.0, 0.0, 0.0, 0.0]
            ])
        cycle_except_third = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0]
            ])
        sequence_table = {
            "s0":
            SequenceGenerator(
                [
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [5, 4, 3, 5, 4, 3, 5, 4, 3],
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [5, 4, 3, 5, 4, 3, 5, 4, 3],
                ]
                , diagonal_transition_matrix(4, 0.8)),
            "s1":
            SequenceGenerator(
                [
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [2, 1, 3, 2, 1, 3, 2, 1, 3],
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [5, 4, 3, 5, 4, 3, 5, 4, 3],
                ]
                , diagonal_transition_matrix(4, 0.8)),
            "s2":
            SequenceGenerator(
                [
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [2, 1, 3, 2, 1, 3, 2, 1, 3],
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [1]
                ]
                , fourth_sequence_excluded),
            "test":
            SequenceGenerator(
                [
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [2, 1, 3, 2, 1, 3, 2, 1, 3],
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [5, 4, 3, 5, 4, 3, 5, 4, 3],
                ]
                , cycle_except_third)
        }

        seqs = [sequence_table[seq] for seq in seq_list]

        if key == "gl_nn":
            model = GlNnTopology(seqs, args.n_predictors)
        elif key == "gl_nn_fg":
            model = GlFakeGateNnTopology()
        else:
            raise NotImplementedError

        model.assign_ids()

        return model

    # Create simulation, it is registers itself to observer_system
    with observer_system_context(storage_file) as observer_system:
        run_topology_with_ui(topology=topology_factory(args.model, args.sequences),
                             seed=seed,
                             max_steps=0,
                             auto_start=False,
                             observer_system=observer_system)

    print('Running simulation, press enter to quit.')
    input()


if __name__ == '__main__':
    main()
