from abc import ABC, abstractmethod

from data_set.domain.interfaces.repositories import DatasetSamplesRepository
from data_set.domain.interfaces.states import ISelectDatasetSamplesState
from duckdi import Interface


@Interface
class ILoadDatasetSamplesFactory(ABC):
    @abstractmethod
    def call(
        self, dataset_samples_repository: DatasetSamplesRepository
    ) -> ISelectDatasetSamplesState: ...
