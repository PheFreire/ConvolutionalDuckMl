from random import shuffle
from typing import List
from modules.datasets.domain.dtos.sample_dto import SampleDto
from modules.neural_network.domain.interfaces.providers import INeuralNetworkProvider
from modules.datasets.domain.interfaces.repositories import IDatasetRepository
from duckdi import Get

class TrainingConvolutionalModelUsecase:
    def __init__(self) -> None:
        self.dataset_repository = Get(IDatasetRepository)
        self.neural_network_provider = Get(INeuralNetworkProvider)

    def execute(self, batch_size: int, n_samples: int, epochs: int) -> None:
        dataset_labels = self.dataset_repository.labels
        batch_per_dataset = batch_size // len(dataset_labels)

        for e in range(1, epochs+1):
            
            for start in range(0, n_samples, batch_per_dataset):
                batch: List[SampleDto] = []

                for dataset in dataset_labels:
                    dataset = self.dataset_repository.get(dataset, start, start+batch_size)
                    batch.extend(dataset.samples())

                shuffle(batch)

                for sample in batch:
                   self.neural_network_provider.propagate(sample.x, sample.y)

        


    
