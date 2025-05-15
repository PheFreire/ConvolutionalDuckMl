from error_function import ErrorFunction
from typing import List, Self
from numpy.typing import NDArray
from layer import Layer
import numpy as np

from layer_setup import LayerSetup

class NeuralNetwork:
    def __init__(self, layers: List[Layer]) -> None:
        self.error_function = ErrorFunction()
        self.layers = layers

    @classmethod
    def new(cls, setup_by_layer: list[LayerSetup], x_shape: list[int]) -> Self:
        input_size = int(np.prod(x_shape)) # Multiply all dimensions leghts

        layers = []
        for layer_setup in setup_by_layer:
            perceptrons_qnt = layer_setup.perceptrons_qnt
            activation_function = layer_setup.activation_function

            layers.append(Layer.new(perceptrons_qnt, input_size, activation_function))
            input_size = perceptrons_qnt

        return cls(layers)

    def propagate(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> np.float64:
        for layer in self.layers:
            x = layer.foward(x)

        return self.error_function.distance(x, y)

    def backpropagate(self, l_rate: np.float64) -> None:
        delta = self.error_function.backpropagate()

        for layer in reversed(self.layers):
            delta = layer.backward(l_rate, delta)
       
