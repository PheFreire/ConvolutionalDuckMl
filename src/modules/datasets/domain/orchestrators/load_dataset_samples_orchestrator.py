from data_set.domain.interfaces.factories import LoadDatasetSamplesFactory
from data_set.domain.interfaces.repositories import DatasetSamplesRepository
from duckdi import Get


class LoadDatasetSamplesOrchestrator:
    def __init__(self) -> None:
        self.dataset_samples_repository = Get(DatasetSamplesRepository)
        self.factory = Get(LoadDatasetSamplesFactory)

    def call(self) -> None:
        select_dataset_samples_state = self.factory.call(
            self.dataset_samples_repository
        )
        serialize_dataset_samples_state = select_dataset_samples_state.call()
        load_samples_into_training_set_state = serialize_dataset_samples_state.call()
        load_samples_into_training_set_state.call()
