from abc import ABC, abstractmethod
from typing import Dict

from modules.hyperparameters.domain.dtos import LayerHyperparameterDto
from modules.neural_network.domain.interfaces.states.i_build_base_activation_function_per_layer_state import \
    IBuildBaseActivationFunctionPerLayerState


class IMapLayerHyperparametersState(ABC):
    @abstractmethod
    def with_hyperparameters(
        self, layer_hyperparameters: Dict[str, LayerHyperparameterDto]
    ) -> IBuildBaseActivationFunctionPerLayerState:
        """
        Load the hyperparameters for each layer in the neural network.

        Parameters:
            layer_hyperparameters (Dict[str, LayerHyperparameterDto]):
                A dictionary containing configuration data for each layer
                (e.g., number of neurons, activation types).

        Returns:
            IBuildBaseActivationFunctionPerLayerState: The next builder state load and setup the activation functions for each layer.
        """
        pass
