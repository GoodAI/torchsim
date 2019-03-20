from typing import List

from torchsim.core.nodes.grayscale_node import GrayscaleNode
from eval_utils import run_just_model
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.nodes.nn_node import NNetParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.nn_node_group import NnNodeGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup


class Task0NnTopology(Topology):
    """ NNet that solves the Task0
    """

    def __init__(self,
                 image_size=SeDatasetSize.SIZE_24,
                 model_seed: int = NNetParams._default_params['seed'],
                 baseline_seed: int = 333,
                 batch_s: int = NNetParams._default_params['batch_size'],
                 buffer_s: int = NNetParams._default_params['buffer_size'],
                 num_cc: List[int] = (20,),
                 lr: int=NNetParams._default_params['lr'],
                 num_epochs: int=NNetParams._default_params['num_epochs'],
                 class_filter=None,
                 random_order: bool=False,
                 experts_on_x: int = 2,
                 use_grayscale=False,
                 use_se=False):
        """
       Constructor of the NN topology which should solve the Task0.
       Args:
           num_cc: number of cluster centers for each layer (including the top one)
           batch_s: batch sizes for each layer
           buffer_s: buffer sizes for each layer
           lr: learning rate for each layer
           experts_on_x: could be used for computing receptive field size in the conv layer, not used now
           model_seed: seed of the model
           image_size: size of the dataset image
           class_filter: filters the classes in the dataset
           baseline_seed: seed for the baseline nodes
       """
        super().__init__('cuda')

        num_channels = 1 if use_grayscale else 3

        self._se_group = SeNodeGroup(baseline_seed=baseline_seed,
                                     layer_sizes=num_cc,
                                     class_filter=class_filter,
                                     image_size=image_size,
                                     random_order=random_order,
                                     use_se=use_se)
        self.add_node(self._se_group)

        self._model = NnNodeGroup(num_labels=20,
                                  image_size=image_size,
                                  num_channels=num_channels,
                                  model_seed=model_seed,
                                  batch_s=batch_s,
                                  buffer_s=buffer_s,
                                  num_epochs=num_epochs,
                                  lr=lr)

        self.add_node(self._model)

        if use_grayscale:
            grayscale_node = GrayscaleNode(squeeze_channel=False)
            self.add_node(grayscale_node)
            Connector.connect(
                self._se_group.outputs.image,
                grayscale_node.inputs.input
            )
            Connector.connect(
                grayscale_node.outputs.output,
                self._model.inputs.image
            )
        else:
            Connector.connect(
                self._se_group.outputs.image,
                self._model.inputs.image
            )

        Connector.connect(
            self._se_group.outputs.labels,
            self._model.inputs.label
        )

    def restart(self):
        pass


if __name__ == '__main__':
    params = [
        {'class_filter': (1, 2, 3, 4)}
    ]

    run_just_model(Task0NnTopology(**params[0]), gui=True)
