from abc import ABC, abstractmethod

from modules.hyperparameters.domain.interfaces.terminals import \
    ILoadHyperTerminal


class IBuildHyperState(ABC):
    @abstractmethod
    def call(self) -> ILoadHyperTerminal:
        """
        # Final state that builds the hyperparameters and loads them into the cache.
            - This state is responsible for constructing the final hyperparameters and returning them for later use in model training or inference.

        # Methods:
            - call: Initiates the return of the hyperparameters.

        # Parameters:
            - None

        # Returns:
            - ILoadHyperTerminal: The final step that return the constructed hyperparameters.
        """
        pass
