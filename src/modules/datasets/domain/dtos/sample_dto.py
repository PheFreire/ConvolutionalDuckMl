from modules.neural_network.domain.interfaces.providers import ITensor
from dataclasses import dataclass

@dataclass
class SampleDto:
    x: ITensor
    y: ITensor
