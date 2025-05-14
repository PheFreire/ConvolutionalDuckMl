from typing import Callable, Dict, Union
import numpy as np
from numpy.typing import NDArray

Numeric = Union[float, np.float64, NDArray[np.float64]]
ActivationFn = Callable[[Numeric], Numeric]

class ActivationFunction:
    def __init__(self) -> None:
        self.activation_functions: Dict[str, Dict[str, ActivationFn]] = {
            'relu': {
                'f': self.relu,
                'b': self.d_relu
            },
            'sigmoid': {
                'f': self.sigmoid,
                'b': self.d_sigmoid
            },
            'softmax': {
                'f': self.softmax,
                'b': self.d_softmax
            },
        }

    def activation_forward(self, activation: str) -> ActivationFn:
        f = self.activation_functions.get(activation, {}).get('f')
        if f is None:
            raise NotImplementedError(f"A função de ativação '{activation}' não foi implementada.")
        return f

    def activation_backward(self, activation: str) -> ActivationFn:
        b = self.activation_functions.get(activation, {}).get('b')
        if b is None:
            raise NotImplementedError(f"A derivada da ativação '{activation}' não foi implementada.")
        return b

    def sigmoid(self, z: Numeric) -> Numeric:
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z: Numeric) -> Numeric:
        s = self.sigmoid(z)
        return s * (1 - s)

    def relu(self, z: Numeric) -> Numeric:
        return np.maximum(0, z)

    def d_relu(self, z: Numeric) -> Numeric:
        return np.where(z > 0, 1.0, 0.0)

    def softmax(self, z: Numeric) -> Numeric:
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def d_softmax(self, z: Numeric) -> Numeric:
        s = self.softmax(z)
        return s * (1 - s)


