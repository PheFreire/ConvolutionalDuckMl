from modules.neural_network.domain.interfaces.providers.i_activation_function_provider import IActivationFunctionProvider
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import ITensor
from abc import abstractmethod
from typing import Self


class ILayerProvider:
    @classmethod
    @abstractmethod
    def new(cls, layer_size: int, input_size: int, activation_function: IActivationFunctionProvider) -> Self:
        """
        Factory method to create a new neural layer.

        Initializes a layer composed of a specified number of perceptrons, each with
        a given input size and a shared activation function.

        Parameters:
            layer_size (int): The number of perceptrons (neurons) in the layer.
            input_size (int): The size of the input vector each perceptron receives.
            activation_function (IActivationFunctionProvider): The activation function to be used
                by all perceptrons in this layer.

        Returns:
            Self: A new instance of the implementing layer class.
        """
        pass

    @abstractmethod
    def foward(self, input: ITensor) -> ITensor:
        """
        Perform the forward pass for the entire layer.

        Applies each perceptron's forward logic to the input tensor and aggregates
        the results into a single output tensor representing the layer's output.

        Parameters:
            input (ITensor): The input tensor to the layer.

        Returns:
            ITensor: The layer's output tensor, composed of all perceptrons' outputs.
        """
        pass

    @abstractmethod
    def backward(self, l_rate: float, delta: ITensor) -> ITensor:
        """
        Perform the backward pass for the layer.

        Propagates the error gradient through each perceptron, updates the weights and biases
        using the learning rate, and calculates the gradient to pass to the previous layer.

        Parameters:
            l_rate (float): The learning rate for gradient descent.
            delta (ITensor): The gradient of the loss with respect to the layer's output.

        Returns:
            ITensor: The gradient of the loss with respect to the layer's input.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of perceptrons (neurons) in the layer.

        Returns:
            int: The number of perceptrons.
        """
        pass

