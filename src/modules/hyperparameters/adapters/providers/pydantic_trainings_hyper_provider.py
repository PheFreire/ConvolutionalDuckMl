from modules.hyperparameters.domain.dtos import TrainingHyperparameterDto, ModelHyperparameterDto
from modules.hyperparameters.domain.interfaces.providers import ITrainingsHyperProvider
from framework.utils import get_field
from typing import Any, Dict


class PydanticTrainingsHyperProvider(ITrainingsHyperProvider):
    def serialize(self, raw_hyper_parameters: Dict[str, Any], model: ModelHyperparameterDto) -> TrainingHyperparameterDto:
        section_data = get_field(self, raw_hyper_parameters, "trainings")
        training = get_field(self, section_data, model.plugged_training)
        return TrainingHyperparameterDto.model_validate(training)
