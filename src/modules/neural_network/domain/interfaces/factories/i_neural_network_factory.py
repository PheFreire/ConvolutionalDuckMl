from abc import ABC, abstractmethod

from duckdi import Interface

from modules.neural_network.domain.interfaces.states import IMapInputLayerState


@Interface
class INeuralNetworkFactory(ABC):
    @abstractmethod
    def start(self) -> IMapInputLayerState:
        """
        Starts the neural network construction pipeline.

        This method initializes the builder chain

        Returns:
            IMapInputLayerState: The next state in the configuration flow, responsible for
                                 mapping the input dimensions into a properly shaped first layer.
        """
        pass
