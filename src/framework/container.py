from duckdi import register

from modules.datasets.adapters.providers.dataset_connection_providers.bin_dataset_connection_provider import BinDatasetConnectionProvider
from modules.datasets.adapters.repositories.dataset_repositories.cache_dataset_repository import CacheDatasetRepository
from modules.hyperparameters.adapters.factories.pydantic_hyper_serializer_factory import PydanticHyperSerializersFactory
from modules.hyperparameters.adapters.providers.toml_hyper_parser_provider import TomlHyperParserProvider
from modules.hyperparameters.adapters.repositories.hyperparameters_repository import HyperparametersRepository
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

register(PydanticHyperSerializersFactory)
register(HyperparametersRepository)
register(TomlHyperParserProvider)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# register(BinDatasetConnectionProvider)
# register(ActivationFunctionProvider)
# register(MseErrorFunctionProvider)
# register(CacheDatasetRepository)
# register(NeuralNetworkProvider)
# register(NeuralNetworkFactory)
# register(PerceptronProvider)
# register(LayerProvider)
# register(NumpyTensor)

