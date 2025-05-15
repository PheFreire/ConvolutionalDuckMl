from abc import ABC, abstractmethod
from typing import Self
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import ITensor

class IActivationFunctionProvider(ABC):
    @classmethod
    @abstractmethod
    def new(cls, activation_function: str) -> Self:
        """
        Instantiate a new activation function provider based on the given function name.

        This factory method initializes the provider using a named activation function
        such as "sigmoid", "relu", or "softmax". The provider will be responsible for
        applying this function and its derivative during forward and backward passes.

        Args:
            activation_function (str): The name of the activation function to use.

        Returns:
            Self: A configured activation function provider instance.
        """
        pass

    @abstractmethod
    def execute(self, input: ITensor) -> ITensor:
        """
        Apply the activation function to the input tensor.

        This method is typically called during the forward pass of a perceptron or layer,
        transforming the pre-activation value (z) into the activated output (ŷ).

        Args:
            input (ITensor): The input tensor to which the activation function will be applied.

        Returns:
            ITensor: The result of applying the activation function element-wise.
        """
        pass

    @abstractmethod
    def d_execute(self, input: ITensor) -> ITensor:
        """
        Compute the derivative of the activation function for the given input.

        This method is used during the backward pass to calculate the local gradient
        of the activation function with respect to the input.

        Args:
            input (ITensor): The input tensor (typically the pre-activation value z).

        Returns:
            ITensor: The element-wise derivative of the activation function evaluated at the input.
        """
        pass
    
    @abstractmethod
    def copy(self) -> Self:
        """
        Create a deep copy of the activation function provider.

        The returned instance should retain the same activation function configuration
        and behavior as the original.

        Returns:
            Self: A new instance of the activation function provider.
        """
        pass

