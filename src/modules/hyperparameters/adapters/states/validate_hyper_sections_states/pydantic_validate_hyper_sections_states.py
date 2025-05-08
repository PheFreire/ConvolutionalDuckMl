from modules.hyperparameters.adapters.states.build_hyper_states import PydanticBuildHyperState
from modules.hyperparameters.domain.interfaces.states import IValidateHyperSectionsState
from modules.hyperparameters.domain.interfaces.states import IBuildHyperState
from modules.hyperparameters.domain.dtos import (
    TrainingHyperparameterDto,
    DatasetHyperparameterDto,
    OutputHyperparameterDto,
    LayerHyperparameterDto,
    ModelHyperparameterDto,
)
from config.app_error import AppError
from typing import Any, Dict, Type
from pydantic import BaseModel

class PydanticValidateHyperSectionsState(IValidateHyperSectionsState):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def call(self) -> IBuildHyperState:
        model = ModelHyperparameterDto.model_validate(self.get(self.config, 'model'))
        dataset = self.validate('datasets', DatasetHyperparameterDto, model.plugged_dataset)
        training = self.validate('trainings', TrainingHyperparameterDto, model.plugged_training)
        output = self.validate('outputs', OutputHyperparameterDto, model.plugged_output)
        layers = self.validate_layers(model)

        return PydanticBuildHyperState(model, dataset, training, output, layers)

    def validate_layers(self, model: ModelHyperparameterDto) -> Dict[str, LayerHyperparameterDto]:
        layers_data = self.get(self.config, 'layers')
        return {
            layer_name: LayerHyperparameterDto.model_validate(self.get(layers_data, layer_name))
            for layer_name in model.plugged_layers
        }

    def validate[T](self, section: str, dto_class: Type[T], sub_key: str) -> T:
        section_data = self.get(self.config, section)
        if isinstance(dto_class, BaseModel):
            return dto_class.model_validate(self.get(section_data, sub_key))

        raise AppError(
            class_pointer=self,
            title="Hyperparameter Validator TypeError", 
            message="Hyperparameter validator should be a Pydantic Basemodel instance!", 
            details={ "invalid_validator_type": type(dto_class) },
            code=500,
        )

    def get(self, struct: dict[str, Any], key: str) -> Any:
        data = struct.get(key)
        if data is not None:
            return data
        
        raise AppError(
            class_pointer=self,
            title="Hyperparameters Validator Key Error", 
            message=f"Hyperparameter key '{key}' not found!", 
            details={"missing_key": key},
            code=500,
        )
