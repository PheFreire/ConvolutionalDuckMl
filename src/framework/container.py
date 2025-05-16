from duckdi import register

from modules.hyperparameters.adapters.factories.load_hyper_factories import \
    TomlLoadHyperFactory
from modules.hyperparameters.adapters.repositories.hyperparameters_repositories.hyperparameters_repository import \
    HyperparametersRepository
from modules.neural_network.adapters.factories.neural_network_factories.neural_network_factory import \
    NeuralNetworkFactory
from modules.neural_network.adapters.providers.activation_function_providers.activation_function_provider import \
    ActivationFunctionProvider
from modules.neural_network.adapters.providers.error_function_providers.mse_error_function_provider import \
    MseErrorFunctionProvider
from modules.neural_network.adapters.providers.layer_providers.layer_provider import \
    LayerProvider
from modules.neural_network.adapters.providers.neural_network_providers.neural_network_provider import \
    NeuralNetworkProvider
from modules.neural_network.adapters.providers.perceptron_providers.perceptron_provider import \
    PerceptronProvider
from modules.neural_network.adapters.providers.tensor_providers.numpy_tensor_provider import \
    NumpyTensor

register(ActivationFunctionProvider)
register(HyperparametersRepository)
register(MseErrorFunctionProvider)
register(NeuralNetworkProvider)
register(NeuralNetworkFactory)
register(TomlLoadHyperFactory)
register(PerceptronProvider)
register(LayerProvider)
register(NumpyTensor)
