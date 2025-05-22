import json
from typing import Dict, Optional, Self

from framework.app_error import AppError
from modules.hyperparameters.domain.dtos import (DatasetHyperparameterDto,
                                                 HyperparametersDto,
                                                 LayerHyperparameterDto,
                                                 ModelHyperparameterDto,
                                                 TrainingHyperparameterDto)
from modules.hyperparameters.domain.dtos.output_hyperparameter_dto import \
    OutputHyperparameterDto
from modules.hyperparameters.domain.interfaces.repositories import \
    IHyperparametersRepository


class HyperparametersRepository(IHyperparametersRepository):
    __hyperparameters: Optional[HyperparametersDto] = None

    @classmethod
    def refresh(cls, hyperparameters: HyperparametersDto) -> Self:
        cls.__hyperparameters = hyperparameters
        return cls()

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

    def __str__(self) -> str:
        line = "-=" * 30

        if self.__hyperparameters is None:
            return ""

        data = json.dumps(
            {
                "model": self.model.model_dump(),
                "dataset": self.dataset.model_dump(),
                "layers": {n: layer.model_dump() for n, layer in self.layers.items()},
                "training": self.training.model_dump(),
                "output": self.output.model_dump(),
            },
            ensure_ascii=False,
            indent=3,
        )

        return f"\n{line}\n\nhyperparameters = {data}\n\n{line}\n"
