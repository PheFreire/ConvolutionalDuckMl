from typing import List

from modules.neural_network.domain.interfaces.providers import (
    IErrorFunctionProvider, ILayerProvider, INeuralNetworkProvider)
from modules.neural_network.domain.interfaces.terminals import \
    INeuralNetworkTerminal


class NeuralNetworkTerminal(INeuralNetworkTerminal):
    def __init__(self, layers: List[ILayerProvider]) -> None:
        self.layers = layers

    def end(
        self,
        neural_network_provider: INeuralNetworkProvider,
        error_function_provider: IErrorFunctionProvider,
    ) -> INeuralNetworkProvider:
        return neural_network_provider.new(self.layers, error_function_provider)
