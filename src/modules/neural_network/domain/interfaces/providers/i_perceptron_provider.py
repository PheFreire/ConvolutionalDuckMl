from modules.neural_network.domain.interfaces.providers.i_activation_function_provider import IActivationFunctionProvider
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import ITensor
from abc import ABC, abstractmethod
from typing import Self

class IPerceptronProvider(ABC):
    @classmethod
    @abstractmethod
    def new(cls, input_size: int, activation_function: IActivationFunctionProvider) -> Self:
        """
        Factory method to create a new perceptron instance.

        This method initializes a perceptron with the given input size and the specified
        activation function provider.

        Parameters:
            input_size (int): The number of input features this perceptron will receive.
            activation_function (IActivationFunctionProvider): An object that provides the
                activation function and its derivative, used in forward and backward passes.

        Returns:
            Self: A new instance of the implementing perceptron class.
        """
        pass

    @abstractmethod
    def foward(self, x: ITensor) -> ITensor:
        """
        Perform the forward pass of the perceptron.

        This method calculates the output of the perceptron given the input tensor `x`.
        It typically includes computing the weighted sum plus bias, followed by applying
        the activation function.

        Parameters:
            x (ITensor): The input tensor.

        Returns:
            ITensor: The output tensor after applying the activation function.
        """
        pass

    @abstractmethod
    def backward(self, delta: ITensor, l_rate: float) -> ITensor:
        """
        Perform the backward pass of the perceptron.

        This method updates the perceptron's weights and bias using the provided
        error gradient (delta) and learning rate. It also computes the gradient to
        propagate to the previous layer in the network.

        Parameters:
            delta (ITensor): The gradient of the loss with respect to the perceptron's output.
            l_rate (float): The learning rate used to scale the weight and bias updates.

        Returns:
            ITensor: The gradient of the loss with respect to the perceptron's input,
                     which should be passed back to the previous layer.
        """
        pass

