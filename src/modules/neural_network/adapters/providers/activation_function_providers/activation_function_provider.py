from typing import Callable, Dict, Self
from modules.neural_network.domain.interfaces.providers import IActivationFunctionProvider, ITensor
from framework.app_error import AppError

class ActivationFunctionProvider(IActivationFunctionProvider):
    def __init__(self, activation_function: str) -> None:
        self.activation_function = activation_function

        self._functions: Dict[str, Dict[str, Callable[[ITensor], ITensor]]] = {
            "sigmoid": {
                "f": self._sigmoid,
                "b": self._d_sigmoid
            },
            "relu": {
                "f": self._relu,
                "b": self._d_relu
            },
            "softmax": {
                "f": self._softmax,
                "b": self._d_softmax
            },
        }

        if activation_function not in self._functions:
            raise AppError(self, "Unknown Activation Function", f"Activation '{activation_function}' is not supported.")

    @classmethod
    def new(cls, activation_function: str) -> Self:
        return cls(activation_function)

    def execute(self, input: ITensor) -> ITensor:
        return self._functions[self.activation_function]["f"](input)

    def d_execute(self, input: ITensor) -> ITensor:
        return self._functions[self.activation_function]["b"](input)

    def copy(self) -> Self:
        return self.new(self.activation_function)

    def _sigmoid(self, z: ITensor) -> ITensor:
        return (1 / (1 + z.exp() * -1))

    def _d_sigmoid(self, z: ITensor) -> ITensor:
        s = self._sigmoid(z)
        return s * (1 - s)

    def _relu(self, z: ITensor) -> ITensor:
        return z.new([val if val > 0 else 0 for val in z])

    def _d_relu(self, z: ITensor) -> ITensor:
        return z.new([1.0 if val > 0 else 0.0 for val in z])

    def _softmax(self, z: ITensor) -> ITensor:
        z_exp = (z - z.max()).exp()
        return z_exp / z_exp.sum()

    def _d_softmax(self, z: ITensor) -> ITensor:
        s = self._softmax(z)
        return s * (1 - s)

