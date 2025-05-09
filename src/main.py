from numpy._typing import NDArray
from typing import Optional
import numpy as np

class Perceptron:
    def __init__(self, input_size: int, num_neurons: int) -> None:
        self.weight = np.random.rand(num_neurons, input_size)
        self.bias = np.random.uniform(-1, 1, size=(num_neurons,))

        self.last_x: Optional[NDArray[np.float64]] = None
        self.last_z: Optional[NDArray[np.float64]] = None

    def foward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.dot(self.weight, x) + self.bias
        self.last_x = x
        self.last_z = z
        return z

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        s = self.sigmoid(z)
        return s * (1 - s)

    def upgrade(self, l_rate: float, delta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.last_x is None or self.last_z is None:
            raise ValueError("Você precisa chamar foward() antes de upgrade().")

        dz = delta * self.d_sigmoid(self.last_z)
        dw = np.outer(dz, self.last_x)

        self.weight -= l_rate * dw
        self.bias -= l_rate * dz
        return dz

