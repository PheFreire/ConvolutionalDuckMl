from modules.hyperparameters.domain.dtos import LayerHyperparameterDto, ModelHyperparameterDto
from modules.hyperparameters.domain.interfaces.providers import ILayerHyperProvider
from framework.utils import get_field
from typing import Any, Dict


class PydanticLayerHyperProvider(ILayerHyperProvider):
    def serialize(self, raw_hyper_parameters: Dict[str, Any], model: ModelHyperparameterDto) -> Dict[str, LayerHyperparameterDto]:
        layers_data = get_field(self, raw_hyper_parameters, "layers")
        return {
            layer_name: LayerHyperparameterDto.model_validate(
                get_field(self, layers_data, layer_name)
            )
            for layer_name in model.plugged_layers
        }
