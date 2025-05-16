from abc import ABC, abstractmethod
from typing import Self

from duckdi import Interface

from modules.neural_network.domain.interfaces.providers.i_tensor_provider import \
    ITensor


@Interface
class IErrorFunctionProvider(ABC):
    @classmethod
    @abstractmethod
    def new(cls) -> Self:
        """
        Instantiate a new error function provider.

        This factory method creates a fresh instance of the error function,
        independent of the specific loss implementation (e.g., MSE, CrossEntropy).

        Returns:
            Self: A new instance of the implementing error function class.
        """
        pass

    @abstractmethod
    def distance(self, predicted_y: ITensor, true_y: ITensor) -> ITensor:
        """
        Compute the scalar loss between predicted and true output tensors.

        This method is responsible for calculating the current loss value
        and storing any internal state necessary for its derivative.

        Args:
            predicted_y (ITensor): The predicted output tensor from the model.
            true_y (ITensor): The expected (ground truth) output tensor.

        Returns:
            ITensor: A one-element tensor containing the scalar loss value.
        """
        pass

    @abstractmethod
    def backpropagate(self) -> ITensor:
        """
        Compute the gradient of the loss with respect to the predicted outputs.

        This method must be called only after `distance()` has been invoked, as it
        relies on internal values (e.g., difference tensors) stored during that computation.

        Returns:
            ITensor: A tensor representing the gradient of the loss function
                     with respect to the model's predicted output.
        """
        pass
