from abc import ABC, abstractmethod
from typing import Dict, Optional

from modules.hyperparameters.domain.dtos import (
    TrainingHyperparameterDto,
    DatasetHyperparameterDto,
    LayerHyperparameterDto,
    ModelHyperparameterDto,
    HyperparametersDto,
)
from modules.hyperparameters.domain.dtos.output_hyperparameter_dto import OutputHyperparameterDto

class IHyperparametersRepository(ABC):
    """
    # Repository for managing hyperparameters.
        - This repository stores the hyperparameters and provides methods to access the dataset, layers, training configurations, and output configurations.
        - The hyperparameters are stored in a private class variable and can be refreshed using the `refresh` method.

    # Attributes:
        - `__hyperparameters` (Optional[HyperparametersDto]): The hyperparameters data.
    """
    __hyperparameters: Optional[HyperparametersDto]

    @classmethod
    @abstractmethod
    def refresh(cls, hyperparameters: HyperparametersDto) -> None:
        """
        # Refresh the hyperparameters stored in the repository.

        # This method updates the internal hyperparameters data with the provided
            - `HyperparametersDto`.

        # Parameters:
            - hyperparameters (HyperparametersDto): The new set of hyperparameters.
        """

    @property
    @abstractmethod
    def model(self) -> ModelHyperparameterDto:
        """
        # Retrieves the model configuration from the stored hyperparameters.

        The `model` property fetches the model-related hyperparameters from the loaded data, such as the 
        plugged dataset, layers, training configuration, and output configuration. It raises an error 
        if the hyperparameters are not loaded.

        # Returns:
            - ModelHyperparameterDto: The model configuration, which contains the dataset, layers, 
              training, and output settings.

        # Raises:
            - AppError: If hyperparameters are not loaded, an AppError is raised with a detailed message.
        """

    @property
    @abstractmethod
    def dataset(self) -> DatasetHyperparameterDto:
        """
        # Retrieves the dataset configuration from the stored hyperparameters.

        # Returns:
            - ModelHyperparameterDto: The dataset configuration.

        # Raises:
            - ValueError: If hyperparameters are not loaded, an error is raised.
        """
    
    @property
    @abstractmethod
    def layers(self) -> Dict[str, LayerHyperparameterDto]:
        """
        # Retrieves the layers configuration from the stored hyperparameters.

        # Returns:
            - Dict[str, LayerHyperparameterDto]: The layers configuration.

        # Raises:
            - ValueError: If hyperparameters are not loaded, an error is raised.
        """

    @property
    @abstractmethod
    def training(self) -> TrainingHyperparameterDto:
        """
        # Retrieves the training configuration from the stored hyperparameters.

        # Returns:
            - TrainingHyperparameterDto: The training configuration.

        # Raises:
            - ValueError: If hyperparameters are not loaded, an error is raised.
        """

    @property
    @abstractmethod
    def output(self) -> OutputHyperparameterDto:
        """
        # Retrieves the output configuration from the stored hyperparameters.

        # Returns:
            - OutputHyperparameterDto: The output configuration.

        # Raises:
            - ValueError: If hyperparameters are not loaded, an error is raised.
        """
