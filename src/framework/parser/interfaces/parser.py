from abc import ABC, abstractmethod
from typing import Any, Dict

from duckdi import Interface


@Interface
class IParser(ABC):
    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]: ...
