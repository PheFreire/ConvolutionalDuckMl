from modules.neural_network.adapters.states.map_layer_hyperparameters_states import \
    MapLayerHyperparametersState
from modules.neural_network.domain.interfaces.providers import ITensor
from modules.neural_network.domain.interfaces.states import (
    IMapInputLayerState, IMapLayerHyperparametersState)


class MapInputLayerState(IMapInputLayerState):
    def with_input(self, x: ITensor) -> IMapLayerHyperparametersState:
        return MapLayerHyperparametersState(type(x), x.flat().shape()[0])
