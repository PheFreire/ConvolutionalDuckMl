from modules.neural_network.domain.interfaces.providers import ITensor
from modules.datasets.domain.dtos import DatasetDto
from abc import ABC, abstractmethod


class IDatasetProvider(ABC):
    @abstractmethod
    def unpack(self, tensor: ITensor) -> DatasetDto:
        pass
