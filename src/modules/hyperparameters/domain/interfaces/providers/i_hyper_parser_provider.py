from abc import ABC, abstractmethod
from typing import Any, Dict
from duckdi import Interface

@Interface
class IHyperParserProvider(ABC):
    @abstractmethod
    def parse(self) -> Dict[str, Any]:
        pass
