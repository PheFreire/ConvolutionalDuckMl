from random import uniform
from typing import Optional, Self

from framework.app_error import AppError
from modules.neural_network.domain.interfaces.providers import (
    IActivationFunctionProvider, IPerceptronProvider, ITensor)


class PerceptronProvider(IPerceptronProvider):
    def __init__(
        self,
        w: Optional[ITensor] = None,
        b: Optional[ITensor] = None,
        activation_function: Optional[IActivationFunctionProvider] = None,
        tensor_type: Optional[ITensor] = None,
    ) -> None:
        self.__tensor_type = tensor_type
        self.__w = w
        self.__b = b
        self.__activation_function = activation_function

        self.last_z: Optional[ITensor] = None
        self.last_x: Optional[ITensor] = None

        if self.__w and len(self.w.shape()) != 1:
            raise AppError(
                self,
                "Perceptron Tensor Shape Error",
                "Weight tensor must be flatten (1D).",
                {"invalid_tensor": self.w},
            )

    @property
    def w(self) -> ITensor:
        if isinstance(self.__w, ITensor):
            return self.__w
        raise AppError(
            self, "Weight Access Error", "Weight tensor not initialized properly."
        )

    @w.setter
    def w(self, value: ITensor) -> None:
        if not isinstance(value, ITensor):
            raise AppError(
                self,
                "Weight Assignment Error",
                "Assigned value is not a valid ITensor.",
            )

        if len(value.shape()) != 1:
            raise AppError(self, "Weight Shape Error", "Weight tensor must be 1D.")
        self.__w = value

    @property
    def b(self) -> ITensor:
        if isinstance(self.__b, ITensor):
            return self.__b
        raise AppError(
            self, "Bias Access Error", "Bias tensor not initialized properly."
        )

    @b.setter
    def b(self, value: ITensor) -> None:
        if not isinstance(value, ITensor):
            raise AppError(
                self, "Bias Assignment Error", "Assigned value is not a valid ITensor."
            )
        self.__b = value

    @property
    def activation_function(self) -> IActivationFunctionProvider:
        if isinstance(self.__activation_function, IActivationFunctionProvider):
            return self.__activation_function
        raise AppError(
            self, "Activation Function Error", "Invalid or missing activation function."
        )

    @property
    def tensor_type(self) -> ITensor:
        if isinstance(self.__tensor_type, ITensor):
            return self.__tensor_type
        raise AppError(self, "Tensor Type Error", "Invalid or missing tensor adapter.")

    @classmethod
    def new(
        cls,
        input_size: int,
        activation_function: IActivationFunctionProvider,
        tensor_type: ITensor,
    ) -> Self:
        w = tensor_type.from_random(input_size)
        b = tensor_type.new([uniform(-1, 1)])
        return cls(w, b, activation_function, tensor_type)

    def forward(self, x: ITensor) -> ITensor:
        self.last_z = self.w.dot(x) + self.b
        self.last_x = x
        return self.activation_function.execute(self.last_z)

    def backward(self, delta: ITensor, l_rate: ITensor) -> ITensor:
        if self.last_x is None or self.last_z is None:
            raise AppError(self, "Forward Missing", "Execute forward before backward.")

        # Local gradient: dz = δ * σ'(z)
        dz = self.activation_function.d_execute(self.last_z) * delta

        # Weight update: w = w - learning_rate * (x * dz)
        self.w -= (self.last_x * dz) * l_rate

        # Bias update
        self.b -= dz * l_rate

        # Return gradient to previous layer
        return self.w * dz

    def copy(self) -> Self:
        return self.new(
            self.w.shape()[0], self.activation_function.copy(), self.tensor_type.copy()
        )
