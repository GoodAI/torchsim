from abc import abstractmethod

import torch
from torchsim.core.graph.node_base import EmptyOutputs
from torchsim.core.graph.node_group import GenericGroupInputs, NodeGroupBase


class ClassificationModelGroupInputs(GenericGroupInputs['ClassificationModelGroup']):
    """Inputs of the group required by the experiment."""

    def __init__(self, owner):
        super().__init__(owner)
        self.image = self.create('Image')
        self.label = self.create('Label')


class ClassificationModelGroup(NodeGroupBase[ClassificationModelGroupInputs, EmptyOutputs]):
    """ This should be implemented by your model in order to support the classification accuracy measurement."""
    def __init__(self, name: str):
        super().__init__(name, inputs=ClassificationModelGroupInputs(self))

    @abstractmethod
    def get_average_log_delta_for(self, layer_id: int) -> float:
        """Average log delta across all dimensions and all the cluster centers. If delta==0, return 0."""
        pass

    @abstractmethod
    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        """Returns a tensor representing the class label predicted by the architecture."""
        pass

    @abstractmethod
    def model_switch_to_training(self):
        """Switch experts (NN) to training"""
        pass

    @abstractmethod
    def model_switch_to_testing(self):
        """Switch experts (NN) to testing"""
        pass
