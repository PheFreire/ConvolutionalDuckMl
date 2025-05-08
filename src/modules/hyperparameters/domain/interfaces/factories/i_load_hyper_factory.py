from modules.hyperparameters.domain.interfaces.states import IReadHyperFileState
from abc import ABC, abstractmethod
from duckdi import Interface

@Interface
class ILoadHyperFactory(ABC):
    @abstractmethod
    def call(self) -> IReadHyperFileState:
        """
        # Factory responsible for initiating the hyperparameter loading process.
            - This factory defines a method to create the first state of the hyperparameter loading chain, which is responsible for reading the hyperparameter file and initiating the validation process.

        # Methods:
            - call: Creates and returns the first state in the loading chain that reads the hyperparameter file.

        # Parameters:
            - None

        # Returns:
            - IReadHyperFileState: The state that reads the hyperparameter file and initiates the validation chain.
        """
        pass
