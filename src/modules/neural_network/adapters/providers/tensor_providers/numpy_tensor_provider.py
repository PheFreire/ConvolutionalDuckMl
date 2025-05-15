from modules.neural_network.domain.interfaces.providers import ITensor
from modules.neural_network.domain.interfaces.providers import Matrix
from typing import Any, Iterator, List, Union, Self 
from numpy.typing import NDArray

import numpy as np

class NumpyTensor(ITensor):
    def __init__(self, data: NDArray[np.float64]=np.array([], dtype=np.float64)):
        self._data = data.astype(np.float64)

    @classmethod
    def new(cls, tensor: Matrix) -> Self:
        return cls(np.array(tensor, dtype=np.float64))

    @classmethod
    def from_random(cls, *shape: int) -> Self:
        return cls(np.random.rand(*shape))

    @classmethod
    def from_tensors(cls, tensors: List[Self]) -> Self:
        data = np.array([t.value for t in tensors], dtype=np.float64)
        return cls(data)

    @property
    def value(self) -> NDArray[np.float64]:
        return self._data


    def __add__(self, other: Union[Self, float]) -> 'NumpyTensor':
        data = self._data + (other._data if isinstance(other, NumpyTensor) else other)
        return NumpyTensor(data)

    def __radd__(self, other: float) -> 'NumpyTensor':
        return self.__add__(other)

    def __iadd__(self, other: Union[Self, float]) -> Self:
        self._data += other._data if isinstance(other, NumpyTensor) else other
        return self


    def __sub__(self, other: Union[Self, float]) -> 'NumpyTensor':
        data = self._data - (other._data if isinstance(other, NumpyTensor) else other)
        return NumpyTensor(data)

    def __rsub__(self, other: float) -> 'NumpyTensor':
        return self.__sub__(other)

    def __isub__(self, other: Union[Self, float]) -> Self:
        self._data -= other._data if isinstance(other, NumpyTensor) else other
        return self


    def __mul__(self, other: Union[Self, float]) -> 'NumpyTensor':
        data = self._data * (other._data if isinstance(other, NumpyTensor) else other)
        return NumpyTensor(data)

    def __rmul__(self, other: float) -> 'NumpyTensor':
        return self.__mul__(other)

    def __imul__(self, other: Union[Self, float]) -> Self:
        self._data *= other._data if isinstance(other, NumpyTensor) else other
        return self


    def __truediv__(self, other: Union[Self, float]) -> 'NumpyTensor':
        data = self._data / (other._data if isinstance(other, NumpyTensor) else other)
        return NumpyTensor(data)

    def __rtruediv__(self, other: float) -> 'NumpyTensor':
        return self.__truediv__(other)

    def __itruediv__(self, other: Union[Self, float]) -> Self:
        self._data /= other._data if isinstance(other, NumpyTensor) else other
        return self


    def __floordiv__(self, other: Union[Self, float]) -> 'NumpyTensor':
        data = self._data // (other._data if isinstance(other, NumpyTensor) else other)
        return NumpyTensor(data)

    def __rfloordiv__(self, other: float) -> 'NumpyTensor':
        return self.__floordiv__(other)

    def __ifloordiv__(self, other: Union[Self, float]) -> Self:
        self._data //= other._data if isinstance(other, NumpyTensor) else other
        return self


    def __pow__(self, other: float) -> 'NumpyTensor':
        data = self._data ** (other._data if isinstance(other, NumpyTensor) else other)
        return NumpyTensor(data)

    def __rpow__(self, other: float) -> 'NumpyTensor':
        return self.__pow__(other)

    def __ipow__(self, other: float) -> Self:
        self._data **= other._data if isinstance(other, NumpyTensor) else other
        return self


    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NumpyTensor):
            return np.array_equal(self._data, other.value)
        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    
    def __iter__(self) -> Iterator[float]:
        return iter(self._data.flatten())


    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def flat(self) -> 'NumpyTensor':
        return NumpyTensor(self._data.flatten())

    def dot(self, other: Self) -> 'NumpyTensor':
        return NumpyTensor(np.dot(self._data, other.value))

    def mean(self) -> float:
        return float(np.mean(self._data))

    def exp(self) -> 'NumpyTensor':
        return NumpyTensor(np.exp(self._data))

    def sum(self) -> float:
        return float(np.sum(self._data))

    def copy(self) -> 'NumpyTensor':
        return NumpyTensor(self._data.copy())
    
    def max(self) -> float:
        return float(np.max(self._data))
