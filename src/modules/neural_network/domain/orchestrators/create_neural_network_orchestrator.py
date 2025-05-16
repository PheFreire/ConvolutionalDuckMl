from duckdi import Get

from modules.hyperparameters.domain.interfaces.repositories import \
    IHyperparametersRepository
from modules.neural_network.domain.interfaces.factories import \
    INeuralNetworkFactory
from modules.neural_network.domain.interfaces.providers import (
    IActivationFunctionProvider, IErrorFunctionProvider, ILayerProvider,
    INeuralNetworkProvider, IPerceptronProvider, ITensor)


class CreateNeuralNetworkOrchestrator:
    """
    Responsible for building a configured neural network instance
    using provided factory, hyperparameters, and architecture providers.
    """

    def __init__(self) -> None:
        self.activation_function_provider = Get(IActivationFunctionProvider)
        self.hyperparameters_repository = Get(IHyperparametersRepository)
        self.neural_network_provider = Get(INeuralNetworkProvider)
        self.error_function_provider = Get(IErrorFunctionProvider)
        self.neural_network_factory = Get(INeuralNetworkFactory)
        self.perceptron_provider = Get(IPerceptronProvider)
        self.layer_provider = Get(ILayerProvider)

    def execute(self, x: ITensor) -> INeuralNetworkProvider:
        """
        Builds and returns a fully configured neural network instance.

        Parameters:
            x (ITensor): Input tensor to define the input shape.

        Returns:
            INeuralNetworkProvider: A complete, ready-to-train neural network instance.
        """

        return (
            self.neural_network_factory.start()
            .with_input(x)
            .with_hyperparameters(self.hyperparameters_repository.layers)
            .with_activation_function(self.activation_function_provider)
            .with_perceptron(self.perceptron_provider)
            .with_layer(self.layer_provider)
            .end(self.neural_network_provider, self.error_function_provider)
        )
