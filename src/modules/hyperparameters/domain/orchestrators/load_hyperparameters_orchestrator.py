from modules.hyperparameters.domain.interfaces.factories import ILoadHyperFactory
from duckdi import Get

class LoadHyperparametersOrchestrator:
    def __init__(self) -> None:
        self.factory = Get(ILoadHyperFactory)

    def execute(self) -> None:
        read_hyper_file_state = self.factory.call()
        validate_hyper_sections_state = read_hyper_file_state.call()
        build_hyper_state = validate_hyper_sections_state.call()
        load_hyper_into_cache_state = build_hyper_state.call()
        load_hyper_into_cache_state.call()


