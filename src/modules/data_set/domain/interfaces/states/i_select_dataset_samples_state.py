from data_set.domain.interfaces.states.i_serialize_dataset_samples_state import ISerializeDatasetSamplesState
from abc import ABC, abstractmethod

class ISelectDatasetSamplesState(ABC):
    @abstractmethod
    def call(self) -> ISerializeDatasetSamplesState: ...
