from abc import abstractmethod

import torch

from torchsim.core.graph.node_base import EmptyOutputs
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.node_groups.classification_model_group import \
    ClassificationModelGroupInputs


# TODO add classification ouputs here
class MultilayerModelGroup(NodeGroupBase[ClassificationModelGroupInputs, EmptyOutputs]):
    """ This should be implemented by your model in order to support the classification accuracy measurement."""
    def __init__(self, name: str):
        super().__init__(name, inputs=ClassificationModelGroupInputs(self))

    # sp statistics
    @abstractmethod
    def get_average_log_delta_for(self, layer_id: int) -> float:
        """Average log delta across all dimensions and all the cluster centers. If delta==0, return 0."""
        pass

    @abstractmethod
    def get_average_boosting_duration_for(self, layer_id: int) -> float:
        pass

    @abstractmethod
    def get_num_boosted_clusters_ratio(self, layer_id: int) -> float:
        pass

    # values used for further computation
    @abstractmethod
    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        """Returns a tensor representing the class label predicted by the architecture."""
        pass

    # layer properties
    @abstractmethod
    def get_flock_size_of(self, layer_id: int) -> int:
        pass

    @abstractmethod
    def get_sp_size_for(self, layer_id: int) -> int:
        pass

    @abstractmethod
    def get_output_id_for(self, layer_id: int) -> int:
        pass

    @abstractmethod
    def clone_sp_output_tensor_for(self, layer_id: int) -> torch.Tensor:
        pass

    # train/test
    @abstractmethod
    def model_switch_to_training(self):
        """Switch experts (NN) to training"""
        pass

    @abstractmethod
    def model_switch_to_testing(self):
        """Switch experts (NN) to testing"""
        pass

    @abstractmethod
    def is_learning(self):
        """Just for debugging, anwer if in learning phase"""
        pass

    @abstractmethod
    def num_layers(selfk):
        """Total number of layers of the model"""
        pass
