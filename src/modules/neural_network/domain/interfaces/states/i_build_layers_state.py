from modules.neural_network.domain.interfaces.terminals import INeuralNetworkTerminal
from modules.neural_network.domain.interfaces.providers import ILayerProvider
from abc import ABC, abstractmethod

class IBuildLayersState(ABC):
    @abstractmethod
    def with_layer(self, layer_provider: ILayerProvider) -> INeuralNetworkTerminal:
        """
        Set the layer provider to be used for constructing complete layers from ther base perceptrons.

        Parameters:
            layer_provider (ILayerProvider): 
                Responsible for building layers composed of perceptrons with the defined hyperparameters.

        Returns:
            INeuralNetworkTerminal: The terminal builder state to finalize the network building.
        """
        pass
