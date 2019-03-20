from eval_utils import parse_test_args, run_just_model
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_conv_wide_topology_more_labels import \
    Task0ConvWideTopologyMoreLabels

if __name__ == '__main__':
    arg = parse_test_args()

    params = [
        {'num_cc': [170, 400, 400],
         'lr': [0.1, 0.3, 0.3],
         'batch_s': [4000, 1500, 1500],
         'buffer_s': [4500, 2000, 2000],
         'label_scale': 1},

    ]

    params2 = [
        {'num_cc': [100, 200, 200],
         'lr': [0.4, 0.4, 0.4],
         'batch_s': [1500, 1000, 1000],
         'buffer_s': [3000, 1500, 1100],
         'label_scale': 1}
    ]

    run_just_model(Task0ConvWideTopologyMoreLabels(**params2[0]), gui=True)
