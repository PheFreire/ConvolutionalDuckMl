from modules.hyperparameters.adapters.providers import (
    PydanticDatasetHyperProvider,
    PydanticLayerHyperProvider,
    PydanticModelHyperProvider,
    PydanticOutputHyperProvider,
    PydanticTrainingsHyperProvider,
)

from modules.hyperparameters.domain.interfaces.factories import IHyperSerializersFactory
from modules.hyperparameters.domain.interfaces.providers import (
    IDatasetHyperProvider,
    ILayerHyperProvider,
    IModelHyperProvider,
    IOutputHyperProvider,
    ITrainingsHyperProvider,
)

class PydanticHyperSerializersFactory(IHyperSerializersFactory):
    @property
    def dataset_hyper_provider(self) -> IDatasetHyperProvider: 
        return PydanticDatasetHyperProvider()
    
    @property
    def layer_hyper_provider(self) -> ILayerHyperProvider: 
        return PydanticLayerHyperProvider()

    @property
    def model_hyper_provider(self) -> IModelHyperProvider: 
        return PydanticModelHyperProvider()
    
    @property
    def output_hyper_provider(self) -> IOutputHyperProvider: 
        return PydanticOutputHyperProvider()
    
    @property
    def trainings_hyper_provider(self) -> ITrainingsHyperProvider: 
        return PydanticTrainingsHyperProvider()
