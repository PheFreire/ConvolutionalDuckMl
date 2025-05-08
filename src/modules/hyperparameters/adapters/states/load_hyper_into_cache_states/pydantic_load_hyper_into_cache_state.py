from modules.hyperparameters.adapters.repositories.hyperparameters_repositories import PydanticHyperparametersRepository
from modules.hyperparameters.domain.interfaces.states import ILoadHyperIntoCacheState
from modules.hyperparameters.domain.dtos import HyperparametersDto

class PydanticLoadHyperIntoCacheState(ILoadHyperIntoCacheState):
    def __init__(self, hyperparameters: HyperparametersDto) -> None:
        self.hyperparameters = hyperparameters

    def call(self) -> None:
        PydanticHyperparametersRepository.refresh(self.hyperparameters)
        
        
        
