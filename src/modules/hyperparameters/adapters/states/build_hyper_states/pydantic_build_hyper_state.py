from modules.hyperparameters.adapters.states.load_hyper_into_cache_states import PydanticLoadHyperIntoCacheState
from modules.hyperparameters.domain.interfaces.states import IBuildHyperState, ILoadHyperIntoCacheState
from modules.hyperparameters.domain.dtos import (
    TrainingHyperparameterDto,
    DatasetHyperparameterDto,
    OutputHyperparameterDto,
    LayerHyperparameterDto,
    ModelHyperparameterDto,
    HyperparametersDto,
)
from typing import Dict

class PydanticBuildHyperState(IBuildHyperState):
    def __init__(
        self,
        training: TrainingHyperparameterDto,
        dataset: DatasetHyperparameterDto,
        output: OutputHyperparameterDto,
        layers: Dict[str, LayerHyperparameterDto],
        model: ModelHyperparameterDto,
    ) -> None:
        self.hyperparameters = {
            "training": training, 
            "dataset": dataset, 
            "output": output, 
            "layers": layers,
            "model": model, 
        }

    def call(self) -> ILoadHyperIntoCacheState:
        return PydanticLoadHyperIntoCacheState(
            HyperparametersDto.model_validate(self.hyperparameters)
        )
