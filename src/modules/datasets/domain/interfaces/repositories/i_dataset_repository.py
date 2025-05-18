from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Self

from modules.datasets.domain.dtos import DatasetDto
from duckdi import Interface


@Interface
class IDatasetRepository(ABC):
    datasets: Dict[str, DatasetDto] = {}

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def all(self) -> List[DatasetDto]:
        pass
    
    @abstractmethod
    def get(self, dataset: str, start:int=0, batch: Optional[int] = None, random: bool = True) -> DatasetDto:
        pass

    @classmethod
    @abstractmethod
    def refresh(cls, datasets: List[DatasetDto]) -> Self:
        pass
