from abc import ABC, abstractmethod

from data_set.domain.interfaces.states.i_serialize_dataset_samples_state import \
    ISerializeDatasetSamplesState


class ISelectDatasetSamplesState(ABC):
    @abstractmethod
    def call(self) -> ISerializeDatasetSamplesState: ...
