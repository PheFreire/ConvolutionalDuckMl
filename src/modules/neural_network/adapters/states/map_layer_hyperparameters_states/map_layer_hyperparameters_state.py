from typing import Dict, List, Type

from modules.hyperparameters.domain.dtos import LayerHyperparameterDto
from modules.neural_network.adapters.states.build_base_activation_function_per_layer_states import \
    BuildBaseActivationFunctionPerLayerState
from modules.neural_network.domain.dtos import LayerSetup
from modules.neural_network.domain.interfaces.providers.i_tensor_provider import \
    ITensor
from modules.neural_network.domain.interfaces.states import (
    IBuildBaseActivationFunctionPerLayerState, IMapLayerHyperparametersState)


class MapLayerHyperparametersState(IMapLayerHyperparametersState):
    def __init__(self, tensor_type: Type[ITensor], x_size: int) -> None:
        self.tensor_type = tensor_type
        self.x_size = x_size

    def with_hyperparameters(
        self, layer_hyperparameters: Dict[str, LayerHyperparameterDto]
    ) -> IBuildBaseActivationFunctionPerLayerState:
        layers_setups: List[LayerSetup] = []

        for hyperparameters in layer_hyperparameters.values():
            layers_setups.append(
                LayerSetup.from_hyperparameters(hyperparameters, self.x_size)
            )
            self.x_size = hyperparameters.num_nodes

        return BuildBaseActivationFunctionPerLayerState(self.tensor_type, layers_setups)
