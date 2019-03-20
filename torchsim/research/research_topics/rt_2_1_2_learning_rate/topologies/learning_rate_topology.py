from typing import Sequence

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.models.expert_params import SamplingMethod
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.conv_wide_two_layer_topology import \
    ConvWideTwoLayerTopology


class LearningRateTopology(ConvWideTwoLayerTopology):
    """The same as the parent, but with potentially different default parameters"""

    def __init__(self,
                 use_dataset: bool = True,
                 image_size=SeDatasetSize.SIZE_24,
                 model_seed: int = 321,
                 baseline_seed: int = 333,
                 num_cc: Sequence[int] = (300, 600),
                 batch_s: Sequence[int] = (3000, 1000),
                 buffer_s: Sequence[int] = (6000, 6000),
                 sampling_m: SamplingMethod = SamplingMethod.BALANCED,
                 cbt: Sequence[int] = (1000, 1000),
                 lr: Sequence[float] = (0.1, 0.2),  # result from the learning_rate experiment
                 mbt: int = 1000,
                 class_filter=None,
                 experts_on_x: int = 2,
                 label_scale: float = 1,
                 seq_len: int = 3):
        super().__init__(
            use_dataset=use_dataset,
            image_size=image_size,
            model_seed=model_seed,
            baseline_seed=baseline_seed,
            num_cc=num_cc,
            batch_s=batch_s,
            buffer_s=buffer_s,
            sampling_m=sampling_m,
            cbt=cbt,
            lr=lr,
            mbt=mbt,
            class_filter=class_filter,
            experts_on_x=experts_on_x,
            label_scale=label_scale,
            seq_len=seq_len
        )
