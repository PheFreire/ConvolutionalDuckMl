from modules.hyperparameters.domain.dtos import LayerHyperparameterDto, ModelHyperparameterDto
from abc import ABC, abstractmethod
from typing import Any, Dict

class ILayerHyperProvider(ABC):
    @abstractmethod
    def serialize(self, raw_hyper_parameters: Dict[str, Any], model: ModelHyperparameterDto) -> Dict[str, LayerHyperparameterDto]:
        pass
