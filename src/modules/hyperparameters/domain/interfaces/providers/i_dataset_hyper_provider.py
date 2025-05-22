from modules.hyperparameters.domain.dtos import DatasetHyperparameterDto, ModelHyperparameterDto
from abc import ABC, abstractmethod
from typing import Any, Dict

class IDatasetHyperProvider(ABC):
    @abstractmethod
    def serialize(self, raw_hyper_parameters: Dict[str, Any], model: ModelHyperparameterDto) -> DatasetHyperparameterDto:
        pass
