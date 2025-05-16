from typing import List

from modules.neural_network.adapters.terminals.neural_network_terminal import \
    NeuralNetworkTerminal
from modules.neural_network.domain.dtos import LayerSetup
from modules.neural_network.domain.interfaces.providers import (
    ILayerProvider, IPerceptronProvider)
from modules.neural_network.domain.interfaces.states import IBuildLayersState
from modules.neural_network.domain.interfaces.terminals import \
    INeuralNetworkTerminal


class BuildLayersState(IBuildLayersState):
    def __init__(
        self,
        layers_setups: List[LayerSetup],
        base_perceptrons: List[IPerceptronProvider],
    ) -> None:
        self.layers_setups = layers_setups
        self.base_perceptrons = base_perceptrons

    def with_layer(self, layer_provider: ILayerProvider) -> INeuralNetworkTerminal:
        layers = [
            layer_provider.new(layer_setup.num_nodes, base_perceptron)
            for layer_setup, base_perceptron in zip(
                self.layers_setups, self.base_perceptrons
            )
        ]

        return NeuralNetworkTerminal(layers)
