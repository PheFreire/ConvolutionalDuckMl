from modules.hyperparameters.domain.dtos import DatasetHyperparameterDto, ModelHyperparameterDto
from modules.hyperparameters.domain.interfaces.providers import IDatasetHyperProvider
from framework.utils import get_field
from typing import Any, Dict


class PydanticDatasetHyperProvider(IDatasetHyperProvider):
    def serialize(self, raw_hyper_parameters: Dict[str, Any], model: ModelHyperparameterDto) -> DatasetHyperparameterDto:
        section_data = get_field(self, raw_hyper_parameters, "datasets")
        dataset = get_field(self, section_data, model.plugged_dataset)
        return DatasetHyperparameterDto.model_validate(dataset)
    
    
