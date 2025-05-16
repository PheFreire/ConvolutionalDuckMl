from typing import Self

from modules.hyperparameters.domain.dtos import LayerHyperparameterDto


class LayerSetup(LayerHyperparameterDto):
    input_size: int

    @classmethod
    def from_hyperparameters(
        cls, layer_hyperparameter_dto: LayerHyperparameterDto, input_size: int
    ) -> Self:
        layer_setup = layer_hyperparameter_dto.model_dump()
        layer_setup["input_size"] = input_size

        return cls(**layer_setup)
