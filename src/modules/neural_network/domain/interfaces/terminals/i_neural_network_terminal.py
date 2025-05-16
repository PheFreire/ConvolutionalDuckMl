from abc import ABC, abstractmethod

from modules.neural_network.domain.interfaces.providers import (
    IErrorFunctionProvider, INeuralNetworkProvider)


class INeuralNetworkTerminal(ABC):
    @abstractmethod
    def end(
        self,
        neural_network_provider: INeuralNetworkProvider,
        error_function_provider: IErrorFunctionProvider,
    ) -> INeuralNetworkProvider:
        """
        Finalize the construction of the neural network and return the configured instance.

        This method completes the builder pattern flow by injecting the neural network
        implementation and the error function provider. It produces a fully configured
        and ready-to-train model.

        Args:
            neural_network_provider (INeuralNetworkProvider): The core neural network logic,
                capable of managing forward and backward passes.
            error_function_provider (IErrorFunctionProvider): The error function used to compute
                the loss and its gradient during training.

        Returns:
            INeuralNetworkProvider: A fully initialized, trainable neural network instance.
        """
        pass
