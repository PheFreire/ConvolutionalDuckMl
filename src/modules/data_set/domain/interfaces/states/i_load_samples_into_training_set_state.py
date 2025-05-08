from abc import ABC, abstractmethod

class ILoadSamplesIntoTrainingSetState(ABC):
    @abstractmethod
    def call(self) -> None: ...
