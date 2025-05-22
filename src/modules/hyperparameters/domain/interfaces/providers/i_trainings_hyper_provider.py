from modules.hyperparameters.domain.dtos import TrainingHyperparameterDto, ModelHyperparameterDto
from abc import ABC, abstractmethod
from typing import Any, Dict

class ITrainingsHyperProvider(ABC):
    @abstractmethod
    def serialize(self, raw_hyper_parameters: Dict[str, Any], model: ModelHyperparameterDto) -> TrainingHyperparameterDto:
        pass
