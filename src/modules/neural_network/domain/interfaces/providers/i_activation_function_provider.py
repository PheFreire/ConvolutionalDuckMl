from modules.neural_network.domain.interfaces.providers.i_tensor_provider import ITensor
from abc import ABC, abstractmethod
from typing import Self

class IActivationFunctionProvider(ABC):
    @classmethod
    @abstractmethod
    def new(cls, action_function: str) -> Self:
        """
        Factory method to create an activation function provider.

        Initializes the provider with a specified activation function by name,
        such as "sigmoid", "relu", or "softmax".

        Parameters:
            action_function (str): The name of the activation function to use.

        Returns:
            Self: An instance of the implementing activation function provider class.
        """
        pass

    @abstractmethod
    def execute(self, input: ITensor) -> ITensor:
        """
        Apply the activation function to the input tensor.

        This method performs the element-wise activation computation, used during
        the forward pass of a neural network.

        Parameters:
            input (ITensor): The input tensor to be activated.

        Returns:
            ITensor: The result after applying the activation function.
        """
        pass

    @abstractmethod
    def d_execute(self, input: ITensor) -> ITensor:
        """
        Compute the derivative of the activation function.

        This method is typically used during the backward pass for computing
        gradients with respect to the output of the activation function.

        Parameters:
            input (ITensor): The tensor with values to differentiate (usually the pre-activation output).

        Returns:
            ITensor: The gradient (derivative) of the activation function evaluated at the input.
        """
        pass

