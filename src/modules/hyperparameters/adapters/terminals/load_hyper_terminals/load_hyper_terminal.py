from modules.hyperparameters.domain.interfaces.terminals import ILoadHyperTerminal
from modules.hyperparameters.domain.dtos import HyperparametersDto

class LoadHyperTerminal(ILoadHyperTerminal):
    def __init__(self, hyperparameters: HyperparametersDto) -> None:
        self.hyperparameters = hyperparameters

    def call(self) -> HyperparametersDto:
        return self.hyperparameters
