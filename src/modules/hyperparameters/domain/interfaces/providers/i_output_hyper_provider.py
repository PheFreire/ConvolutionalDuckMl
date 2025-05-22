from modules.hyperparameters.domain.dtos import OutputHyperparameterDto, ModelHyperparameterDto
from typing import Any, Dict
from abc import ABC, abstractmethod

class IOutputHyperProvider(ABC):
    @abstractmethod
    def serialize(self, raw_hyper_parameters: Dict[str, Any], model: ModelHyperparameterDto) -> OutputHyperparameterDto:
        pass
