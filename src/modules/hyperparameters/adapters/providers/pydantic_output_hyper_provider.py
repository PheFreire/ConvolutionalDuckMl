from modules.hyperparameters.domain.dtos import OutputHyperparameterDto, ModelHyperparameterDto
from modules.hyperparameters.domain.interfaces.providers import IOutputHyperProvider
from framework.utils import get_field
from typing import Any, Dict


class PydanticOutputHyperProvider(IOutputHyperProvider):
    def serialize(self, raw_hyper_parameters: Dict[str, Any], model: ModelHyperparameterDto) -> OutputHyperparameterDto:
        section_data = get_field(self, raw_hyper_parameters, "outputs")
        output = get_field(self, section_data, model.plugged_output)
        return OutputHyperparameterDto.model_validate(output)
