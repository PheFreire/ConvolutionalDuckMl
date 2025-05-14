from dataclasses import dataclass
from typing import Literal

@dataclass
class LayerSetup:
    perceptrons_qnt: int
    activation_function: Literal['relu', 'softmax', 'sigmoid']
