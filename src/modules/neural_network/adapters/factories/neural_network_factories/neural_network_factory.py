from modules.neural_network.adapters.states.map_input_layer_states.map_input_layer_state import \
    MapInputLayerState
from modules.neural_network.domain.interfaces.factories import \
    INeuralNetworkFactory
from modules.neural_network.domain.interfaces.states import IMapInputLayerState


class NeuralNetworkFactory(INeuralNetworkFactory):
    def start(self) -> IMapInputLayerState:
        return MapInputLayerState()
