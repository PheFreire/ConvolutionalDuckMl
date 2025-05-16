from typing import Callable, Dict, Optional, Self

from framework.app_error import AppError
from modules.neural_network.domain.interfaces.providers import (
    IActivationFunctionProvider, ITensor)


class ActivationFunctionProvider(IActivationFunctionProvider):
    """
    Activation function handler that dynamically applies a specified activation function
    and its derivative during forward and backward passes in a neural network.
    """

    def __init__(self, activation_function: Optional[str] = None) -> None:
        self.activation_function: Optional[str] = activation_function

        self._functions: Dict[str, Dict[str, Callable[[ITensor], ITensor]]] = {
            "sigmoid": {"f": self._sigmoid, "b": self._d_sigmoid},
            "relu": {"f": self._relu, "b": self._d_relu},
            "softmax": {"f": self._softmax, "b": self._d_softmax},
        }

        if (
            activation_function is not None
            and activation_function not in self._functions
        ):
            raise AppError(
                self,
                "Unknown Activation Function",
                f"Activation '{activation_function}' is not supported.",
            )

    @classmethod
    def new(cls, activation_function: str) -> Self:
        """
        Create a new instance configured with a specific activation function.

        Args:
            activation_function (str): Name of the activation function ("relu", "sigmoid", "softmax").

        Returns:
            Self: A configured ActivationFunctionProvider instance.
        """
        return cls(activation_function)

    def execute(self, input: ITensor) -> ITensor:
        """
        Apply the activation function to the input tensor.

        Args:
            input (ITensor): Input tensor to be transformed.

        Returns:
            ITensor: Activated output tensor.
        """
        return self._functions[self.activation]["f"](input)

    def d_execute(self, input: ITensor) -> ITensor:
        """
        Apply the derivative of the activation function to the input tensor.

        Args:
            input (ITensor): Input tensor representing pre-activation or output values.

        Returns:
            ITensor: Gradient of the activation function.
        """
        return self._functions[self.activation]["b"](input)

    def copy(self) -> Self:
        """
        Create a copy of this activation function provider.

        Returns:
            Self: A new instance with the same configuration.
        """
        return self.new(self.activation)

    @property
    def activation(self) -> str:
        """
        Get the name of the configured activation function.

        Returns:
            str: The activation function name.

        Raises:
            AppError: If the activation function was not initialized.
        """
        if isinstance(self.activation_function, str):
            return self.activation_function

        raise AppError(
            self, "Missing Activation Function", "The activation function was not set."
        )

    # --- Internal activation implementations ---

    def _sigmoid(self, z: ITensor) -> ITensor:
        return 1 / (1 + z.exp() * -1)

    def _d_sigmoid(self, z: ITensor) -> ITensor:
        s = self._sigmoid(z)
        return s * (1 - s)

    def _relu(self, z: ITensor) -> ITensor:
        return z.new([val if val > 0 else 0 for val in z])

    def _d_relu(self, z: ITensor) -> ITensor:
        return z.new([1.0 if val > 0 else 0.0 for val in z])

    def _softmax(self, z: ITensor) -> ITensor:
        z_exp = (z - z.max()).exp()
        return z_exp / z_exp.sum()

    def _d_softmax(self, z: ITensor) -> ITensor:
        s = self._softmax(z)
        return s * (1 - s)
