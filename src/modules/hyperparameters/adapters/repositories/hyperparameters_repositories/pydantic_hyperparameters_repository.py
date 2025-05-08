from config.app_error import AppError
from modules.hyperparameters.domain.dtos.output_hyperparameter_dto import OutputHyperparameterDto
from modules.hyperparameters.domain.interfaces.repositories import IHyperparametersRepository
from modules.hyperparameters.domain.dtos import (
    TrainingHyperparameterDto,
    DatasetHyperparameterDto,
    LayerHyperparameterDto,
    ModelHyperparameterDto,
    HyperparametersDto,
)
from typing import Dict, Optional

class PydanticHyperparametersRepository(IHyperparametersRepository):
    __hyperparameters: Optional[HyperparametersDto] = None
    
    @classmethod
    def refresh(cls, hyperparameters: HyperparametersDto) -> None:
        cls.__hyperparameters = hyperparameters

    @property
    def model(self) -> ModelHyperparameterDto:
        return self.__hyper.model

    @property
    def dataset(self) -> DatasetHyperparameterDto:
        return self.__hyper.dataset
    
    @property
    def layers(self) -> Dict[str, LayerHyperparameterDto]:
        return self.__hyper.layers

    @property
    def training(self) -> TrainingHyperparameterDto:
        return self.__hyper.training

    @property
    def output(self) -> OutputHyperparameterDto:
        return self.__hyper.output
    
    @property
    def __hyper(self) -> HyperparametersDto:
        if self.__hyperparameters is not None:
            return self.__hyperparameters
        
        raise AppError(
            class_pointer=self,
            title="Hyperparameters Not Loaded",
            message="The hyperparameters have not been loaded yet. Please load the hyperparameters first.",
            details={"current_state": "not_loaded"},
            code=400,
        )
