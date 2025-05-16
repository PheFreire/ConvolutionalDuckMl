from typing import List, Optional, Self

from framework.app_error import AppError
from modules.neural_network.domain.interfaces.providers import (
    ILayerProvider, IPerceptronProvider)
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import \
    ITensor


class LayerProvider(ILayerProvider):
    def __init__(self, perceptrons: Optional[List[IPerceptronProvider]] = None) -> None:
        self.__perceptrons = perceptrons

    @property
    def perceptrons(self) -> List[IPerceptronProvider]:
        if isinstance(self.__perceptrons, list) and all(
            isinstance(p, IPerceptronProvider) for p in self.__perceptrons
        ):
            return self.__perceptrons

        raise AppError(
            self,
            "Invalid Perceptron List",
            "Layer perceptrons were not properly initialized or contain invalid types.",
            {"invalid_perceptrons": self.__perceptrons},
        )

    @classmethod
    def new(cls, layer_size: int, base_perceptron: IPerceptronProvider) -> Self:
        layers = [base_perceptron.copy() for _ in range(layer_size)]
        return cls(layers)

    def forward(self, input: ITensor) -> ITensor:
        tensor_type = type(input)
        outputs = [p.forward(input) for p in self.perceptrons]
        return tensor_type.from_tensors(outputs)

    def backward(self, l_rate: ITensor, delta: ITensor) -> ITensor:
        tensor_type = type(delta)
        gradients = [
            p.backward(tensor_type.new([d]), l_rate)
            for p, d in zip(self.perceptrons, delta)
        ]
        return tensor_type.from_tensors(gradients)

    def __len__(self) -> int:
        return len(self.perceptrons)
