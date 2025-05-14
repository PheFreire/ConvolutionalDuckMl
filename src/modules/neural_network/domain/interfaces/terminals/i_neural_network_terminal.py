from abc import ABC, abstractmethod

from modules.neural_network.domain.interfaces.providers.i_neural_network_provider import INeuralNetworkProvider

class INeuralNetworkTerminal(ABC):
    @abstractmethod
    def end(self) -> INeuralNetworkProvider:
        """
        Finalize the neural network configuration and build the complete model.

        Returns:
            INeuralNetworkProvider: A ready-to-use neural network instance.
        """
        pass
