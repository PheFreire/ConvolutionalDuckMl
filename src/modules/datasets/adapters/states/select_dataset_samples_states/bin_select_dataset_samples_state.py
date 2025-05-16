from data_set.domain.interfaces.states.select_dataset_samples_state import \
    SelectDatasetSamplesState
from data_set.domain.interfaces.states.serialize_dataset_samples_state import \
    SerializeDatasetSamplesState


class BinSelectDatasetSamplesState(SelectDatasetSamplesState):
    def __init__(self, dataset_samples: list[str]) -> None:
        self.dataset_samples = dataset_samples

    def call(
        self, hyperparameters_repository: HyperparametersRepository
    ) -> SerializeDatasetSamplesState:
        pass
