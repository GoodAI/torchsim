from torchsim.research.research_topics.rt_3_1_lr_subfields.node_groups import CN_C1_R1
from torchsim.research.research_topics.rt_3_1_lr_subfields.conv_subfield_layer import SubLrfSpConvLayer, SubLrfConvLayer


class SFCN_C1_R1(CN_C1_R1):
    conv_layer_class = SubLrfConvLayer

    def __init__(self, name: str = None, bottom_layer_size=5, l_0_cluster_centers=10,
                 l_1_cluster_centers=20, l_0_rf_dims=(3, 3), l_0_rf_stride=None, l_1_rf_dims=(2, 2), env_size=(24, 24),
                 label_length=3, sp_n_cluster_centers=10, l_0_sub_field_size=6):
        super().__init__(name, bottom_layer_size, l_0_cluster_centers, l_1_cluster_centers, l_0_rf_dims,
                         l_0_rf_stride, l_1_rf_dims, env_size, label_length, sp_n_cluster_centers,
                         l_0_custom_args={'sub_field_size': l_0_sub_field_size})


class SFSCN_SC1_R1(SFCN_C1_R1):
    conv_layer_class = SubLrfSpConvLayer
