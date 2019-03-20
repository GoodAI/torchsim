from abc import ABC, abstractmethod

import torch


class ModelClassificationAdapterBase(ABC):
    """ This should be implemented by your model in order to support the classification accuracy measurement.
    """

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

    @abstractmethod
    def model_is_learning(self):
        """Return true if the model is learning (debug purposes)"""
        pass
