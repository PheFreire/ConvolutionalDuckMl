from modules.datasets.domain.dtos.sample_dto import SampleDto
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

@dataclass
class DatasetDto:
    samples: Callable[[], Iterable[SampleDto]]
    dataset: str

    def __iter__(self) -> Iterator[SampleDto]:
        """
        Allows the DatasetDto to be used directly in loops.
        
        Returns:
            Iterator[SampleDto]: An iterator over the dataset samples.
        """
        return iter(self.samples())
