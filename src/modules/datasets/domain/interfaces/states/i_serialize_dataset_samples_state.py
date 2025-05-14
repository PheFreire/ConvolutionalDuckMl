from data_set.domain.interfaces.states.i_load_samples_into_training_set_state import ILoadSamplesIntoTrainingSetState
from abc import ABC, abstractmethod

class ISerializeDatasetSamplesState(ABC):
    @abstractmethod
    def call(self) -> ILoadSamplesIntoTrainingSetState: ...
