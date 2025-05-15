from modules.neural_network.domain.interfaces.providers import (
    IActivationFunctionProvider, 
    IPerceptronProvider, 
    ITensor,
)

from framework.app_error import AppError
from typing import Self, Optional
from random import uniform

class PerceptronProvider(IPerceptronProvider):
    def __init__(self, w: ITensor, b: ITensor, activation_function: IActivationFunctionProvider, tensor_type: ITensor) -> None:
        self.tensor_type = tensor_type
        if len(w.shape()) != 1:
            raise AppError(
                self, 
                "Perceptron Tensor Shape Error", "Weight tensor must be flatten (1D).",
                { "invalid_tensor": w }
            )

        self.w = w
        self.b = b
        self.activation_function = activation_function

        self.last_z: Optional[ITensor] = None
        self.last_x: Optional[ITensor] = None

    @classmethod
    def new(cls, input_size: int, activation_function: IActivationFunctionProvider, tensor_type: ITensor) -> Self:
        w = tensor_type.from_random(input_size)
        b = tensor_type.new([uniform(-1, 1)])
        
        return cls(w, b, activation_function, tensor_type)
        
    def forward(self, x: ITensor) -> ITensor:
        self.last_z = self.w.dot(x) + self.b
        self.last_x = x
        
        return self.activation_function.execute(self.last_z)

    def backward(self, delta: ITensor, l_rate: ITensor) -> ITensor:
        if self.last_x is None or self.last_z is None:
            raise AppError(self, "Execute forward before upgrade.")
        
        # Local gradient: dz = δ * σ'(z)
        dz = self.activation_function.d_execute(self.last_z) * delta

        # Weight update: w = w - learning_rate * (x * dz)
        self.w -= (self.last_x * dz) * l_rate

        # Bias update
        self.b -= dz * l_rate
        
        # Return gradient to previous layer
        return self.w * dz

    def copy(self) -> Self:
        return self.new(self.w.shape()[0], self.activation_function.copy(), self.tensor_type.copy())

