from duckdi import Get

from modules.hyperparameters.domain.interfaces.factories import \
    ILoadHyperFactory
from modules.hyperparameters.domain.interfaces.repositories import \
    IHyperparametersRepository


class LoadHyperparametersOrchestrator:
    def __init__(self) -> None:
        self.hyperparameters_repository = Get(IHyperparametersRepository)
        self.load_hyper_factory = Get(ILoadHyperFactory)

    def execute(self) -> IHyperparametersRepository:
        read_hyper_file_state = self.load_hyper_factory.call()
        validate_hyper_sections_state = read_hyper_file_state.call()
        build_hyper_state = validate_hyper_sections_state.call()
        load_hyper_terminal = build_hyper_state.call()
        hyperparameters_dto = load_hyper_terminal.call()

        self.hyperparameters_repository.refresh(hyperparameters_dto)
        return self.hyperparameters_repository
