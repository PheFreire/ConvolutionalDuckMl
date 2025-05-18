from modules.datasets.domain.interfaces.repositories import IDatasetRepository
from modules.datasets.domain.dtos import DatasetDto
from typing import Dict, List, Optional, Self
from framework.utils import get_random
from itertools import islice
from duckdi import Interface

@Interface
class CacheDatasetRepository(IDatasetRepository):
    datasets: Dict[str, DatasetDto] = {}
    
    @property
    def labels(self) -> List[str]:
        return list(self.datasets.keys())

    @property
    def all(self) -> List[DatasetDto]:
        return list(self.datasets.values())
 
    def get(self, dataset: str, start:int=0, batch: Optional[int] = None, random: bool = True) -> DatasetDto:
        batch = start + batch if isinstance(batch, int) else None
        samples = list(islice(self.datasets[dataset].samples(), start, batch))

        if random:
            samples = get_random(samples)

        return DatasetDto(lambda: iter(samples), dataset)

    @classmethod
    def refresh(cls, datasets: List[DatasetDto]) -> Self:
        cls.datasets = {dt.dataset: dt for dt in datasets}
        return cls()
