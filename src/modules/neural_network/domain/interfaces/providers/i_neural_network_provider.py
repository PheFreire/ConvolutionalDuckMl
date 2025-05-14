from modules.neural_network.domain.interfaces.providers import ITensor, ILayerProvider
from abc import ABC, abstractmethod
from typing import Dict, Self

class INeuralNetworkProvider(ABC):
    @abstractmethod
    @classmethod
    def new(cls, layers: Dict[str, ILayerProvider]) -> Self:
        """
        Create a new instance of a neural network using the specified layers.

        Parameters:
            layers (ILayerProvider): A fully constructed set of layers to use in the network.

        Returns:
            Self: An initialized instance of the neural network.
        """
        pass

    @abstractmethod
    def propagate(self, x: ITensor, y: ITensor) -> ITensor:
        """
        Perform the forward pass of the network and compute the loss.

        Parameters:
            x (ITensor): The input tensor.
            y (ITensor): The expected output tensor (labels or targets).
        """
        pass
    
    @abstractmethod
    def back_propagate(self, l_rate: float) -> None:
        """
        Perform the backward pass (gradient computation and update).

        Parameters:
            l_rate (float): The learning rate to be used for weight and bias updates.
        """
        pass
