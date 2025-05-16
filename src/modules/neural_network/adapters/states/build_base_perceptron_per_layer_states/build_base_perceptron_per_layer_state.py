from typing import List, Type

from modules.neural_network.adapters.states.build_layers_states import \
    BuildLayersState
from modules.neural_network.domain.dtos import LayerSetup
from modules.neural_network.domain.interfaces.providers import (
    IActivationFunctionProvider, IPerceptronProvider, ITensor)
from modules.neural_network.domain.interfaces.states import \
    IBuildBasePerceptronPerLayerState
from modules.neural_network.domain.interfaces.states.i_build_layers_state import \
    IBuildLayersState


class BuildBasePerceptronPerLayerState(IBuildBasePerceptronPerLayerState):
    def __init__(
        self,
        tensor_type: Type[ITensor],
        layers_setups: List[LayerSetup],
        activation_functions: List[IActivationFunctionProvider],
    ) -> None:
        self.activation_functions = activation_functions
        self.layers_setups = layers_setups
        self.tensor_type = tensor_type

    def with_perceptron(
        self, perceptron_provider: IPerceptronProvider
    ) -> IBuildLayersState:
        base_perceptrons = [
            perceptron_provider.new(
                layer_setup.input_size, activation_function, self.tensor_type()
            )
            for layer_setup, activation_function in zip(
                self.layers_setups, self.activation_functions
            )
        ]

        return BuildLayersState(self.layers_setups, base_perceptrons)
