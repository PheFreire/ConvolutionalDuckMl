from numpy.typing import NDArray
from framework.app_error import AppError
from perceptron import Perceptron
from typing import List, Self
import numpy as np

class Layer:
    def __init__(self, perceptrons: List[Perceptron]) -> None:
        self.perceptrons = perceptrons
    
    @classmethod
    def new(cls, perceptrons_qnt: int, input_size: int, activation_function: str) -> Self:
        return cls([Perceptron(input_size, activation_function) for _ in range(0, perceptrons_qnt)])
    
    def foward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([n.forward(input) for n in self.perceptrons])

    def backward(self, l_rate: np.float64, delta: NDArray[np.float64]) -> NDArray[np.float64]:
        if (delta.ndim != 1 or len(delta) != len(self.perceptrons)): 
            raise AppError(
                self, 
                "Delta Shape Compatibility Error", 
                f"Delta must be 1D with {len(self.perceptrons)} elements",
                {"invalidDelta": delta}
            )

        return np.array([p.backward(l_rate, d) for p, d in zip(self.perceptrons, delta)])
                
    def __len__(self) -> int:
        return len(self.perceptrons)
