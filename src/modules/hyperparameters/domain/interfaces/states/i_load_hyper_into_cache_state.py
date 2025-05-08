from abc import ABC, abstractmethod

class ILoadHyperIntoCacheState(ABC):
    @abstractmethod
    def call(self) -> None:
        """
        # Final state to load hyperparameters into the cache.
            - This state loads the constructed hyperparameters into the cache so they can be used for further training or inference.

        # Methods:
            - call: Loads the hyperparameters into the cache.

        # Parameters:
            - None

        # Returns:
            - None: No return value, as the hyperparameters are loaded into the cache manager by HyperparametersRepository.
        """
        pass

