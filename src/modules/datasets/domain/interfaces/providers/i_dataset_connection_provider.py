from modules.datasets.domain.interfaces.providers.i_dataset_provider import IDatasetProvider
from modules.hyperparameters.domain.dtos import DatasetHyperparameterDto
from abc import ABC, abstractmethod
from duckdi import Interface
from typing import List

@Interface
class IDatasetConnectionProvider(ABC):
    @abstractmethod
    def connect(self, dataset_hyperparameter_dto: DatasetHyperparameterDto) -> List[IDatasetProvider]:
        pass
