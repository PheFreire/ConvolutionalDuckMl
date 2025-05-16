from typing import List, Optional, Self

from framework.app_error import AppError
from modules.neural_network.domain.interfaces.providers import (
    IErrorFunctionProvider, ILayerProvider, INeuralNetworkProvider, ITensor)


class NeuralNetworkProvider(INeuralNetworkProvider):
    def __init__(
        self,
        layers: Optional[List[ILayerProvider]] = None,
        error_function: Optional[IErrorFunctionProvider] = None,
    ) -> None:
        self.__error_function = error_function
        self.__layers = layers

    @property
    def error_function(self) -> IErrorFunctionProvider:
        if isinstance(self.__error_function, IErrorFunctionProvider):
            return self.__error_function

        raise AppError(
            self,
            "Invalid Error Function",
            "The error function must be an instance of IErrorFunctionProvider.",
        )

    @property
    def layers(self) -> List[ILayerProvider]:
        if self.__layers is not None:
            return self.__layers

        raise AppError(
            self, "Missing Layers", "The network layers were not properly initialized."
        )

    @classmethod
    def new(
        cls, layers: List[ILayerProvider], error_function: IErrorFunctionProvider
    ) -> Self:
        return cls(layers, error_function)

    def propagate(self, x: ITensor, y: ITensor) -> ITensor:
        for layer in self.layers:
            x = layer.forward(x)

        return self.error_function.distance(x, y)

    def back_propagate(self, l_rate: ITensor) -> None:
        delta = self.error_function.backpropagate()

        for layer in reversed(self.layers):
            delta = layer.backward(l_rate, delta)
