from argparse import ArgumentParser

from eval_utils import create_non_persisting_observer_system, run_topology_with_ui
from torchsim.research.se_tasks.topologies.se_task0_basic_topology import SeT0BasicTopology
from torchsim.research.se_tasks.topologies.se_task0_convolutionalSP_topology import SeT0ConvSPTopology
from torchsim.research.se_tasks.topologies.se_task0_convolutional_expert_topology import SeT0ConvTopology
from torchsim.research.se_tasks.topologies.se_task0_narrow_hierarchy import SeT0NarrowHierarchy
from torchsim.research.se_tasks.topologies.se_task1_basic_topology import SeT1Bt
from torchsim.research.se_tasks.topologies.se_task1_conv_topology import SeT1ConvTopologicalGraph


def run_task0(use_dataset: bool):
    bt = SeT0BasicTopology(use_dataset=use_dataset)
    run_topology_with_ui(bt, create_non_persisting_observer_system())


def run_task0_narrow(use_dataset: bool):
    topology = SeT0NarrowHierarchy(use_dataset=use_dataset)
    run_topology_with_ui(topology, create_non_persisting_observer_system())


def run_task0_convSP(use_dataset: bool):
    topology = SeT0ConvSPTopology(use_dataset=use_dataset)
    run_topology_with_ui(topology)


def run_task0_conv(use_dataset: bool, save_gpu_memory: bool):
    topology = SeT0ConvTopology(use_dataset=use_dataset, save_gpu_memory=save_gpu_memory)
    run_topology_with_ui(topology)


def run_task1():
    bt = SeT1Bt()
    run_topology_with_ui(bt)


def run_task1_conv():
    bt = SeT1ConvTopologicalGraph()
    run_topology_with_ui(bt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--topology", choices=['0_basic', '0_narrow', '0_conv', '0_wide', '1_basic', '1_conv'], default='0_basic')
    parser.add_argument("--dataset", action="store_true", default=False)
    parser.add_argument("--save-gpu-memory", action="store_true", default=False)
    args = parser.parse_args()
    _ = {
        '0_basic': lambda: run_task0(args.dataset),
        '0_narrow': lambda: run_task0_narrow(args.dataset),
        '0_conv': lambda: run_task0_convSP(args.dataset),
        '0_wide': lambda: run_task0_conv(args.dataset, args.save_gpu_memory),
        '1_basic': lambda: run_task1(),
        '1_conv': lambda: run_task1_conv()
    }[args.topology]()
