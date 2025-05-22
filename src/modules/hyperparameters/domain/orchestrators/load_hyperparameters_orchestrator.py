from duckdi import Get

from modules.hyperparameters.domain.interfaces.providers.i_hyper_parser_provider import IHyperParserProvider
from modules.hyperparameters.domain.dtos.hyperparameters_dto import HyperparametersDto
from modules.hyperparameters.domain.interfaces.factories import IHyperSerializersFactory
from modules.hyperparameters.domain.interfaces.repositories import \
    IHyperparametersRepository


class LoadHyperparametersOrchestrator:
    def __init__(self) -> None:
        self.hyper_parser_provider = Get(IHyperParserProvider)
        self.hyper_serializers_factory = Get(IHyperSerializersFactory)
        self.hyperparameters_repository = Get(IHyperparametersRepository)

    def execute(self) -> IHyperparametersRepository:
        raw = self.hyper_parser_provider.parse()
        
        model = self.hyper_serializers_factory.model_hyper_provider.serialize(raw)
        dataset = self.hyper_serializers_factory.dataset_hyper_provider.serialize(raw, model) 
        training = self.hyper_serializers_factory.trainings_hyper_provider.serialize(raw, model)
        output = self.hyper_serializers_factory.output_hyper_provider.serialize(raw, model)
        layers = self.hyper_serializers_factory.layer_hyper_provider.serialize(raw, model)

        return self.hyperparameters_repository.refresh(
            HyperparametersDto(
                training=training,
                dataset=dataset,
                output=output,
                layers=layers,
                model=model,
            )
        )
