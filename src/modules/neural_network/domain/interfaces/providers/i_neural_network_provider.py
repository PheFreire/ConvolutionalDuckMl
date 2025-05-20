from abc import ABC, abstractmethod
from typing import List, Self

from duckdi import Interface

from modules.neural_network.domain.interfaces.providers.i_error_function_provider import \
    IErrorFunctionProvider
from modules.neural_network.domain.interfaces.providers.i_layer_provider import \
    ILayerProvider
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import \
    ITensor


@Interface
class INeuralNetworkProvider(ABC):
    @property
    @abstractmethod
    def layers(self) -> List[ILayerProvider]:
        """
        Get the list of layers that compose the neural network.

        This property provides access to the internal ordered list of layers,
        from input to output, that define the structure and depth of the model.

        Returns:
            List[ILayerProvider]: A list of all layers within the network.
        """
        pass

    @property
    @abstractmethod
    def error_function(self) -> IErrorFunctionProvider:
        """
        Get the error (loss) function used by the neural network.

        This function is responsible for computing the loss between predicted
        and expected outputs during forward propagation, as well as computing
        gradients during backpropagation.

        Returns:
            IErrorFunctionProvider: The error function implementation.
        """
        pass

    @classmethod
    @abstractmethod
    def new(
        cls, layers: List[ILayerProvider], error_function: IErrorFunctionProvider
    ) -> Self:
        """
        Instantiate a new neural network using the given ordered layers and error function.

        This factory method builds the network architecture with the provided sequential list of layers.
        The layers are applied in the order they appear in the list. An error function implementation
        is used to compute both the loss during forward propagation and the gradient during backward propagation.

        Args:
            layers (List[ILayerProvider]): A list of neural network layers, ordered from input to output.
            error_function (IErrorFunctionProvider): The error function used for loss calculation and gradient computation.

        Returns:
            Self: A fully initialized neural network instance.
        """
        pass

    @abstractmethod
    def propagate(self, x: ITensor) -> ITensor:
        """
        Perform the forward pass through all layers of the network.

        Each layer receives the output of the previous one, culminating in a final prediction.

        Args:
            x (ITensor): Input tensor representing the input features.
            y (ITensor): Ground-truth output tensor (expected labels or values).

        Returns:
            ITensor: A tensor containing the propagate result.
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
