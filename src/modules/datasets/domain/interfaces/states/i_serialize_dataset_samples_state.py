from abc import ABC, abstractmethod

from data_set.domain.interfaces.states.i_load_samples_into_training_set_state import \
    ILoadSamplesIntoTrainingSetState


class ISerializeDatasetSamplesState(ABC):
    @abstractmethod
    def call(self) -> ILoadSamplesIntoTrainingSetState: ...
