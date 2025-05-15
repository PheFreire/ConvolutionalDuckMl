from modules.neural_network.domain.interfaces.providers import INeuralNetworkProvider, IErrorFunctionProvider, ILayerProvider, ITensor
from typing import Dict, Self

class NeuralNetworkProvider(INeuralNetworkProvider):
    def __init__(self, layers: Dict[str, ILayerProvider], error_function: IErrorFunctionProvider) -> None:
        self.error_function = error_function
        self.layers = layers

    @classmethod
    def new(cls, layers: Dict[str, ILayerProvider], error_function: IErrorFunctionProvider) -> Self:
        return cls(layers, error_function)

    def propagate(self, x: ITensor, y: ITensor) -> ITensor:
        for layer in self.layers.values():
            x = layer.forward(x)

        return self.error_function.distance(x, y)

    def back_propagate(self, l_rate: ITensor) -> None:
        delta = self.error_function.backpropagate()

        for layer in reversed(list(self.layers.values())):
            delta = layer.backward(l_rate, delta)
