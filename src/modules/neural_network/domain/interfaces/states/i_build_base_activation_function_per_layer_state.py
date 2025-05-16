from abc import ABC, abstractmethod

from modules.neural_network.domain.interfaces.providers.i_activation_function_provider import \
    IActivationFunctionProvider
from modules.neural_network.domain.interfaces.states.i_build_base_perceptron_per_layer_state import \
    IBuildBasePerceptronPerLayerState


class IBuildBaseActivationFunctionPerLayerState(ABC):
    @abstractmethod
    def with_activation_function(
        self, activation_function_provider: IActivationFunctionProvider
    ) -> IBuildBasePerceptronPerLayerState:
        """
        Set the activation function provider for each layer in the network.

        Parameters:
            activation_function_provider (IActivationFunctionProvider):
                Provides activation and derivative functions for use in forward/backward passes.

        Returns:
            IBuildBasePerceptronPerLayerState: The next builder state to configure the base perceptrons for each layer.
        """
        pass
