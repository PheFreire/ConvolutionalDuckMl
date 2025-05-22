from framework.utils.network_ui import network_ui
from framework import container
from modules.datasets.domain.usecases.load_dataset_usecase import LoadDatasetUsecase
from modules.hyperparameters.domain.usecases import \
    LoadHyperparametersUsecase
from modules.neural_network.adapters.providers.tensor_providers.numpy_tensor_provider import \
    NumpyTensor
from modules.neural_network.domain.orchestrators.create_neural_network_orchestrator import \
    CreateNeuralNetworkOrchestrator

x = NumpyTensor.new([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

load_hyperparameters_orchestrator = LoadHyperparametersUsecase()
hyper_repository = load_hyperparameters_orchestrator.execute()

# create_neural_network_orchestrator = CreateNeuralNetworkOrchestrator()
# network = create_neural_network_orchestrator.execute(x)

# # print(network_ui(network))

# load_dataset_usecase = LoadDatasetUsecase()
# repository = load_dataset_usecase.execute()

breakpoint()

