import numpy as np
from numpy._typing import NDArray
from typing import Optional

from activation_function import ActivationFunction
from framework.app_error import AppError

# Todo perceptron retorna obrigatoriamente apenas 1 valor escalar (nao matriz) no foward
# idependente da dimensao de x portanto e recomendado fazer um flat na matriz x e que os pesos sejam um matriz 1D de mesmo tamanho que o x "flat"

class Perceptron:
    def __init__(self, input_size: int, activation: str) -> None:
        self.input_size = input_size
        self.weight = np.random.randn(self.input_size)
        self.bias: float = np.random.uniform(-1, 1)

        self.foward_activation = ActivationFunction().activation_forward(activation)
        self.backend_activation = ActivationFunction().activation_forward(activation)

        self.last_x: Optional[NDArray[np.float64]] = None
        self.last_z: Optional[np.float64] = None

    def forward(self, x: NDArray[np.float64]) -> np.float64:
        if x.ndim != 1:
            raise AppError(self, 'Matrix Shape Error', 'matrix should be flat')

        z = np.dot(self.weight, x) + self.bias
        activation = self.foward_activation(z)

        self.last_x = x
        self.last_z = z

        if isinstance(activation, np.float64):
            return activation

        raise AppError(self, 'Foward Activation Response Type Error')

    def backward(self, l_rate: float, delta: np.float64) -> np.float64:
        if self.last_x is None or self.last_z is None:
            raise AppError(self, "Execute forward before upgrade.")
        
        activation_delta = self.backend_activation(self.last_z)

        if isinstance(activation_delta, np.float64):
            dz = delta * activation_delta
            dw = dz * self.last_x

            self.weight -= l_rate * dw
            self.bias -= l_rate * dz
            return dz
        
        raise AppError(self, 'Backward Activation Response Type Error')

