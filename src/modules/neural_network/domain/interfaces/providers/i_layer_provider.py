from abc import abstractmethod
from typing import List, Self

from duckdi import Interface

from modules.neural_network.domain.interfaces.providers.i_perceptron_provider import \
    IPerceptronProvider
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import \
    ITensor


@Interface
class ILayerProvider:
    @property
    @abstractmethod
    def perceptrons(self) -> List[IPerceptronProvider]:
        pass

    @classmethod
    @abstractmethod
    def new(cls, layer_size: int, base_perceptron: IPerceptronProvider) -> Self:
        """
        Instantiate a new neural layer composed of multiple perceptrons.

        This factory method constructs a layer by replicating a given base perceptron
        a number of times equal to `layer_size`. Each perceptron will inherit the same
        configuration (weights, activation function, tensor backend) from the base instance.

        Args:
            layer_size (int): The total number of perceptrons to include in the layer.
            base_perceptron (IPerceptronProvider): A perceptron used as the template for duplication.

        Returns:
            Self: A new instance of the layer, composed of independent perceptrons.
        """
        pass

    @abstractmethod
    def forward(self, input: ITensor) -> ITensor:
        """
        Perform the forward propagation step across all perceptrons in the layer.

        Each perceptron receives the same input tensor and produces a scalar output.
        The outputs are aggregated into a single tensor representing the layer's output vector.

        Args:
            input (ITensor): Input tensor representing the features for the entire layer.

        Returns:
            ITensor: A tensor containing the outputs of all perceptrons in the layer.
        """
        pass

    @abstractmethod
    def backward(self, l_rate: ITensor, delta: ITensor) -> ITensor:
        """
        Perform the backward propagation step across all perceptrons in the layer.

        Each perceptron receives a corresponding slice of the `delta` tensor and uses it
        to update its weights and bias. The method returns the accumulated error to be
        propagated to the previous layer.

        Args:
            l_rate (ITensor): Learning rate tensor used to scale the gradient updates.
            delta (ITensor): Error tensor received from the next layer or loss function,
                with one value per perceptron.

        Returns:
            ITensor: A tensor containing the gradient of the loss with respect to the input.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of perceptrons in the layer.

        Returns:
            int: The total number of perceptrons (neurons) in this layer.
        """
        pass
