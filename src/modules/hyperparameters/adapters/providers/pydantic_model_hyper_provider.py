from modules.hyperparameters.domain.interfaces.providers import IModelHyperProvider
from modules.hyperparameters.domain.dtos import ModelHyperparameterDto
from framework.utils import get_field
from typing import Any, Dict

class PydanticModelHyperProvider(IModelHyperProvider):
    def serialize(self, raw_hyper_parameters: Dict[str, Any]) -> ModelHyperparameterDto:
        raw_model = get_field(self, raw_hyper_parameters, "model")
        return ModelHyperparameterDto.model_validate(raw_model)
