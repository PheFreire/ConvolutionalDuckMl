from modules.neural_network.domain.interfaces.providers import ILayerProvider, IPerceptronProvider
from typing import List, Self

from modules.neural_network.domain.interfaces.providers.i_tensor_provider import ITensor

class LayerProvider(ILayerProvider):
    def __init__(self, perceptrons: List[IPerceptronProvider]) -> None:
        self.perceptrons = perceptrons

    @classmethod
    def new(cls, layer_size: int, base_perceptron: IPerceptronProvider) -> Self:
        layers = [base_perceptron.copy() for _ in range(0, layer_size)]
        return cls(layers)

    def forward(self, input: ITensor) -> ITensor:
        tensor_type = type(input)
        return tensor_type.from_tensors([n.forward(input) for n in self.perceptrons])

    def backward(self, l_rate: ITensor, delta: ITensor) -> ITensor:
        tensor_type = type(delta)
        return tensor_type.from_tensors([p.backward(tensor_type.new([d]), l_rate) for p, d in zip(self.perceptrons, delta)])
