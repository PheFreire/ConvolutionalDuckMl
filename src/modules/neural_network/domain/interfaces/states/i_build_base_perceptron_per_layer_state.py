from modules.neural_network.domain.interfaces.states.i_build_layers_state import IBuildLayersState
from modules.neural_network.domain.interfaces.providers import IPerceptronProvider
from abc import ABC, abstractmethod

class IBuildBasePerceptronPerLayerState(ABC):
    @abstractmethod
    def with_perceptron(self, perceptron_provider: IPerceptronProvider) -> IBuildLayersState:
        """
        Set the a default start perceptron provider to be used for constructing the neurons for each layer.

        Parameters:
            perceptron_provider (IPerceptronProvider): 
                Responsible for creating individual perceptrons with activation and update logic.

        Returns:
            IBuildLayersState: The next builder state to configure the full layers.
        """
        pass
