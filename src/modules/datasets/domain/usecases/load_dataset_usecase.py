from modules.datasets.domain.interfaces.providers.i_dataset_connection_provider import IDatasetConnectionProvider
from modules.datasets.domain.interfaces.repositories import IDatasetRepository
from duckdi import Get

from modules.hyperparameters.domain.interfaces.repositories import IHyperparametersRepository
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import ITensor

class LoadDatasetUsecase:
    def __init__(self) -> None:
        self.dataset_connection_provider = Get(IDatasetConnectionProvider)
        self.hyperparameters_repository = Get(IHyperparametersRepository)
        self.dataset_repository = Get(IDatasetRepository)
        self.tensor = Get(ITensor)

    def execute(self) -> IDatasetRepository: 
        dataset_providers = (
            self.dataset_connection_provider.connect(self.hyperparameters_repository.dataset)
        )
        
        return self.dataset_repository.refresh(
            [dataset_provider.unpack(self.tensor) for dataset_provider in dataset_providers]
        )
