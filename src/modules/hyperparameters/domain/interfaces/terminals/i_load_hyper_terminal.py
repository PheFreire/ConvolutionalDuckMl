from abc import ABC, abstractmethod

from modules.hyperparameters.domain.dtos import HyperparametersDto


class ILoadHyperTerminal(ABC):
    @abstractmethod
    def call(self) -> HyperparametersDto:
        """
        # Final state to load and return hyperparameters.
            - This state return the constructed hyperparameters so they can be used for further training or inference.

        # Methods:
            - call: return the loaded hyperparameters.

        # Parameters:
            - None

        # Returns:
            - HyperparametersDto: loaded hyperparameters.
        """
        pass
