from data_set.adapters.states.select_dataset_samples_states.bin_select_dataset_samples_state import BinSelectDatasetSamplesState
from data_set.domain.interfaces.factories.load_dataset_samples_factory import LoadDatasetSamplesFactory
from data_set.domain.interfaces.repositories import DatasetSamplesRepository
from data_set.domain.interfaces.states import SelectDatasetSamplesState

class BinLoadDatasetSamplesFactory(LoadDatasetSamplesFactory):
    def call(self, dataset_samples_repository: DatasetSamplesRepository) -> SelectDatasetSamplesState:
        return BinSelectDatasetSamplesState(dataset_samples_repository.find())

