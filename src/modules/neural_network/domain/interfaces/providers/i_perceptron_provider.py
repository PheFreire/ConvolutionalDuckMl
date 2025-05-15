from modules.neural_network.domain.interfaces.providers.i_activation_function_provider import IActivationFunctionProvider
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import ITensor
from abc import ABC, abstractmethod
from typing import Self
from duckdi import Interface

@Interface
class IPerceptronProvider(ABC):
    @classmethod
    @abstractmethod
    def new(cls, input_size: int, activation_function: IActivationFunctionProvider, tensor_type: ITensor) -> Self:
        """
        Create a new perceptron instance with initialized weights and bias.

        This factory method constructs a perceptron configured to receive a specific number
        of input features. The perceptron uses the provided activation function and tensor
        adapter to initialize and manage internal computations.

        Args:
            input_size (int): Number of input features expected by this perceptron.
            activation_function (IActivationFunctionProvider): Provider responsible for computing
                the activation and its derivative during forward and backward passes.
            tensor_type (ITensor): A tensor adapter used to initialize and perform tensor operations.
                Determines the numerical backend (e.g., NumPy, Torch).

        Returns:
            Self: A fully initialized perceptron instance.
        """
        pass

    @abstractmethod
    def forward(self, x: ITensor) -> ITensor:
        """
        Perform the forward computation for the perceptron.

        This method calculates the perceptron's output given the input tensor `x`.
        The process involves computing a weighted sum followed by applying the
        activation function.

        Args:
            x (ITensor): Input tensor representing the features for this perceptron.

        Returns:
            ITensor: Output tensor after applying weights, bias, and activation function.
        """
        pass

    @abstractmethod
    def backward(self, delta: ITensor, l_rate: ITensor) -> ITensor:
        """
        Perform the backward computation (backpropagation) for the perceptron.

        This method updates the perceptron's weights and bias based on the error gradient `delta`,
        and the provided learning rate. It also computes the gradient to propagate to the previous
        layer of the network.

        Args:
            delta (ITensor): The error signal from the next layer or loss function.
            l_rate (ITensor): Learning rate tensor used to scale updates.

        Returns:
            ITensor: The propagated gradient with respect to the perceptron's input.
        """
        pass
    
    @abstractmethod
    def copy(self) -> Self:
        """
        Create a deep copy of this perceptron.

        The copied instance should retain the same weights, bias, activation function, and tensor type.
        This is typically used when duplicating perceptrons to build layers.

        Returns:
            Self: A new instance identical to the current perceptron.
        """
        pass

