from modules.hyperparameters.domain.interfaces.states.i_build_hyper_state import IBuildHyperState
from modules.hyperparameters.domain.interfaces.terminals.i_load_hyper_terminal import ILoadHyperTerminal
from modules.hyperparameters.adapters.terminals.load_hyper_terminals.load_hyper_terminal import LoadHyperTerminal
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
        model: ModelHyperparameterDto,
        dataset: DatasetHyperparameterDto,
        training: TrainingHyperparameterDto,
        output: OutputHyperparameterDto,
        layers: Dict[str, LayerHyperparameterDto],
    ) -> None:
        self.hyperparameters = {
            "training": training, 
            "dataset": dataset, 
            "output": output, 
            "layers": layers,
            "model": model, 
        }

    def call(self) -> ILoadHyperTerminal:
        return LoadHyperTerminal(
            HyperparametersDto.model_validate(self.hyperparameters)
        )
