from modules.neural_network.domain.interfaces.providers import ITensor, ILayerProvider, IErrorFunctionProvider
from abc import ABC, abstractmethod
from typing import Dict, Self

class INeuralNetworkProvider(ABC):
    @classmethod
    @abstractmethod
    def new(cls, layers: Dict[str, ILayerProvider], error_function: IErrorFunctionProvider) -> Self:
        """
        Create a new neural network instance with defined layers and error function.

        This factory method constructs the neural network using the provided mapping
        of labeled layers (e.g., "input", "hidden_1", "output") and an error function
        to compute the loss and its gradient.

        Args:
            layers (Dict[str, ILayerProvider]): A dictionary mapping layer names to their implementations.
            error_function (IErrorFunctionProvider): The error function used to compute the loss and its gradient.

        Returns:
            Self: A fully initialized instance of the neural network.
        """
        pass

    @abstractmethod
    def propagate(self, x: ITensor, y: ITensor) -> ITensor:
        """
        Perform the forward pass through all layers of the network and compute the loss.

        Each layer receives the output of the previous one, culminating in a final prediction.
        The loss between the final output and the target is computed using the provided error function.

        Args:
            x (ITensor): Input tensor representing the input features.
            y (ITensor): Ground-truth output tensor (expected labels or values).

        Returns:
            ITensor: A tensor containing the scalar loss value.
        """
        pass

    @abstractmethod
    def back_propagate(self, l_rate: ITensor) -> None:
        """
        Execute the backward pass to compute gradients and update all trainable parameters.

        This method uses the previously computed loss (from `propagate`) to initiate backpropagation,
        updating weights and biases using the specified learning rate.

        Args:
            l_rate (ITensor): A tensor representing the learning rate to apply during parameter updates.
        """
        pass

