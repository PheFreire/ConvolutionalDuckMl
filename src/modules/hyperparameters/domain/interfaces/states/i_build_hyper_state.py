from modules.hyperparameters.domain.interfaces.states.i_load_hyper_into_cache_state import ILoadHyperIntoCacheState
from abc import ABC, abstractmethod

class IBuildHyperState(ABC):
    @abstractmethod
    def call(self) -> ILoadHyperIntoCacheState:
        """
        # Final state that builds the hyperparameters and loads them into the cache.
            - This state is responsible for constructing the final hyperparameters and loading them into the cache for later use in model training or inference.

        # Methods:
            - call: Initiates the loading of the hyperparameters into the cache.

        # Parameters:
            - None

        # Returns:
            - ILoadHyperIntoCacheState: The final step that loads the constructed hyperparameters into the cache.
        """
        pass

