from abc import ABC, abstractmethod
from duckdi import Interface

@Interface
class DatasetSamplesRepository(ABC):
    @abstractmethod
    def find(self) -> list[str]: ...

