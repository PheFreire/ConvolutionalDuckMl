from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Self, TypeAlias, Union

from duckdi import Interface

Matrix: TypeAlias = Union[float, list["Matrix"]]


@Interface
class ITensor(ABC):
    @classmethod
    @abstractmethod
    def new(cls, tensor: Matrix) -> Self:
        """
        Create a tensor from a recursive structure of nested lists or scalars.
        Example: [[1, 2], [3, 4]]
        """
        pass

    @classmethod
    @abstractmethod
    def from_random(cls, *shape: int) -> Self:
        """
        Generate a tensor filled with random values within the given shape.
        Example: random(2, 3) → tensor with shape (2, 3)
        """
        pass

    @classmethod
    @abstractmethod
    def zeros(cls, *shape: int) -> Self:
        """
        Generate a tensor filled with zeros within the given shape.
        Example: zeros(2, 3) → tensor with shape (2, 3)
        """
        pass

    @classmethod
    @abstractmethod
    def from_tensors(cls, tensors: List[Self]) -> Self:
        """
        Construct a single tensor from a list of tensors.

        This is typically used to combine the outputs of multiple perceptrons
        (e.g., in a layer) into a single tensor vector. All input tensors must
        be compatible (e.g., scalars or 1D tensors of the same shape).

        Parameters:
            tensors (List[Self]): A list of individual tensor instances to combine.

        Returns:
            Self: A new tensor containing the concatenated values.
        """
        pass

    @property
    @abstractmethod
    def value(self) -> Any:
        """
        Return the internal raw tensor value (NumPy array, PyTorch tensor, etc.).
        """
        pass

    @abstractmethod
    def __add__(self, other: Union[Self, float]) -> Self: ...
    @abstractmethod
    def __radd__(self, other: float) -> Self: ...
    @abstractmethod
    def __iadd__(self, other: Union[Self, float]) -> Self: ...

    @abstractmethod
    def __sub__(self, other: Union[Self, float]) -> Self: ...
    @abstractmethod
    def __rsub__(self, other: float) -> Self: ...
    @abstractmethod
    def __isub__(self, other: Union[Self, float]) -> Self: ...

    @abstractmethod
    def __mul__(self, other: Union[Self, float]) -> Self: ...
    @abstractmethod
    def __rmul__(self, other: float) -> Self: ...
    @abstractmethod
    def __imul__(self, other: Union[Self, float]) -> Self: ...

    @abstractmethod
    def __truediv__(self, other: Union[Self, float]) -> Self: ...
    @abstractmethod
    def __rtruediv__(self, other: float) -> Self: ...
    @abstractmethod
    def __itruediv__(self, other: Union[Self, float]) -> Self: ...

    @abstractmethod
    def __floordiv__(self, other: Union[Self, float]) -> Self: ...
    @abstractmethod
    def __rfloordiv__(self, other: float) -> Self: ...
    @abstractmethod
    def __ifloordiv__(self, other: Union[Self, float]) -> Self: ...

    @abstractmethod
    def __pow__(self, exponent: float) -> Self: ...
    @abstractmethod
    def __rpow__(self, base: float) -> Self: ...
    @abstractmethod
    def __ipow__(self, exponent: float) -> Self: ...

    @abstractmethod
    def __eq__(self, other: Any) -> bool: ...
    @abstractmethod
    def __ne__(self, other: Any) -> bool: ...

    @abstractmethod
    def __iter__(self) -> Iterator[float]: ...

    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Return the tensor's shape as a tuple of integers.
        Example: (2, 3)
        """
        pass

    @abstractmethod
    def flat(self) -> Self:
        """
        Return a 1D (flattened) version of the tensor.
        Example: tensor with shape (2, 3) → shape (6,)
        """
        pass

    @abstractmethod
    def dot(self, other: Self) -> Self:
        """
        Perform a dot product (inner product) between this tensor and another.
        Shapes must be compatible for matrix multiplication.
        """
        pass

    @abstractmethod
    def mean(self) -> float:
        """
        Compute the mean of all tensor elements.
        """
        pass

    @abstractmethod
    def exp(self) -> Self:
        """
        Apply the exponential function element-wise.
        Returns a new tensor where each element is e^x.
        """
        pass

    @abstractmethod
    def sum(self) -> float:
        """
        Return the sum of all tensor elements.
        """
        pass

    @abstractmethod
    def copy(self) -> Self:
        """
        Return a new tensor that is a deep copy of this one.
        Modifications to the copy will not affect the original.
        """
        pass

    @abstractmethod
    def max(self) -> float:
        """
        Return the maximum value in the tensor.

        Returns:
            float: The largest scalar value in the tensor.
        """
        pass
