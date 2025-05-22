from modules.hyperparameters.domain.dtos import ModelHyperparameterDto
from abc import ABC, abstractmethod
from typing import Any, Dict

class IModelHyperProvider(ABC):
    @abstractmethod
    def serialize(self, raw_hyper_parameters: Dict[str, Any]) -> ModelHyperparameterDto:
        pass
