from duckdi import Interface
from modules.hyperparameters.domain.interfaces.providers import (
    IDatasetHyperProvider,
    ILayerHyperProvider,
    IModelHyperProvider,
    IOutputHyperProvider,
    ITrainingsHyperProvider,
)
from abc import ABC, abstractmethod

@Interface
class IHyperSerializersFactory(ABC):
    @property
    @abstractmethod
    def dataset_hyper_provider(self) -> IDatasetHyperProvider:
        pass
    
    @property
    @abstractmethod
    def layer_hyper_provider(self) -> ILayerHyperProvider:
        pass

    @property
    @abstractmethod
    def model_hyper_provider(self) -> IModelHyperProvider:
        pass
    
    @property
    @abstractmethod
    def output_hyper_provider(self) -> IOutputHyperProvider:
        pass
    
    @property
    @abstractmethod    
    def trainings_hyper_provider(self) -> ITrainingsHyperProvider:
        pass
