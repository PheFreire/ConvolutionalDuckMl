from modules.neural_network.domain.interfaces.states.i_map_layer_hyperparameters_state import IMapLayerHyperparametersState
from modules.neural_network.domain.interfaces.providers import ITensor
from abc import ABC, abstractmethod

class IMapInputLayerState(ABC):
    @abstractmethod
    def with_input(self, x: ITensor) -> IMapLayerHyperparametersState:
        """
        Maps the input tensor's dimensions to construct the first layer of the neural network.

        This method extracts the lenght of the flatten input tensor and uses it to determine the correct
        configuration for the input layer, ensuring compatibility with the subsequent layers.

        Parameters:
            x (ITensor): The tensor representing the input data sample.

        Returns:
            IMapLayerHyperparametersState: The next builder state for defining the network's
                                            hidden and output layers.
        """
        pass
