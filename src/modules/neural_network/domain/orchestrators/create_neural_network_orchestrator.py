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
        # (Map Input Layer)
        tensor_type = type(x)
        x_size = x.flat().shape()[0]


        # (Map Layer Hyperparameters)
        layers_setups: List[LayerSetup] = []

        for hyperparameters in self.hyperparameters_repository.layers.values():
            layers_setups.append(
                LayerSetup.from_hyperparameters(hyperparameters, self.x_size)
            )
            self.x_size = hyperparameters.num_nodes


        # (BuildBaseActivationFunctionPerLayer)
        activation_function_provider = self.activation_function_provider
        
        activation_functions = [
            activation_function_provider.new(layer_setup.activation)
            for layer_setup in self.layers_setups
        ]


        # (BuildBasePerceptronPerLayerState)
        perceptron_provider = self.perceptron_provider
        
        base_perceptrons = [
            perceptron_provider.new(
                layer_setup.input_size, activation_function, self.tensor_type()
            )
            for layer_setup, activation_function in zip(
                self.layers_setups, self.activation_functions
            )
        ]

        # (BuildLayersState)
        layer_provider = self.layer_provider 

        layers = [
            layer_provider.new(layer_setup.num_nodes, base_perceptron)
            for layer_setup, base_perceptron in zip(
                self.layers_setups, self.base_perceptrons
            )
        ]

        # (NeuralNetworkTerminal)

        neural_network_provider = self.neural_network_provider
        error_function_provider = self.error_function_provider
        return neural_network_provider.new(self.layers, error_function_provider)
