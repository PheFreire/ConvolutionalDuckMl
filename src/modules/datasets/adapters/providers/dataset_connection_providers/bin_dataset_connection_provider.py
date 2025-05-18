from modules.datasets.adapters.providers.dataset_providers.bin_dataset_provider import BinDatasetProvider
from modules.datasets.domain.interfaces.providers import IDatasetConnectionProvider, IDatasetProvider
from modules.hyperparameters.domain.dtos import DatasetHyperparameterDto
from framework.app_error import AppError
from typing import List
import os

class BinDatasetConnectionProvider(IDatasetConnectionProvider):
    def connect(self, dataset_hyperparameter_dto: DatasetHyperparameterDto) -> List[IDatasetProvider]:
        datasets_path = dataset_hyperparameter_dto.address

        if not os.path.isdir(datasets_path):
            raise AppError(
            self, 
            'Dataset Directory Not Found Error', 
            f'Could not find dataset directory on "{datasets_path}"',
        )
        
        dataset_providers = []
        for idx, dataset in enumerate(dataset_hyperparameter_dto.samples):
            bin_path = os.path.join(datasets_path, dataset.name)

            if not bin_path.endswith('.bin'):
                raise AppError(
                    self, 
                    'Dataset File Not Found Error', 
                    f'The dataset {dataset.name} should be a path to a ".bin" file',
                    { "invalidPath": bin_path },
                    500,
                )
            
            dataset_buffer = open(bin_path, "rb")
            dataset.name = dataset.name.replace('.bin', '')
            dataset_providers.append(BinDatasetProvider(dataset_buffer, dataset, float(idx)))
        
        return dataset_providers
