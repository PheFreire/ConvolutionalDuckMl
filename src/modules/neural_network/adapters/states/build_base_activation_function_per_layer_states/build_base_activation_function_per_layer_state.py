from typing import List, Type

from modules.neural_network.adapters.states.build_base_perceptron_per_layer_states import \
    BuildBasePerceptronPerLayerState
from modules.neural_network.domain.dtos import LayerSetup
from modules.neural_network.domain.interfaces.providers import \
    IActivationFunctionProvider
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import \
    ITensor
from modules.neural_network.domain.interfaces.states import (
    IBuildBaseActivationFunctionPerLayerState,
    IBuildBasePerceptronPerLayerState)


class BuildBaseActivationFunctionPerLayerState(
    IBuildBaseActivationFunctionPerLayerState
):
    def __init__(
        self, tensor_type: Type[ITensor], layers_setups: List[LayerSetup]
    ) -> None:
        self.layers_setups = layers_setups
        self.tensor_type = tensor_type

    def with_activation_function(
        self, activation_function_provider: IActivationFunctionProvider
    ) -> IBuildBasePerceptronPerLayerState:
        activation_functions = [
            activation_function_provider.new(layer_setup.activation)
            for layer_setup in self.layers_setups
        ]

        return BuildBasePerceptronPerLayerState(
            self.tensor_type,
            self.layers_setups,
            activation_functions,
        )
