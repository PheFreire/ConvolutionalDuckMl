from pydantic import BaseModel, Field
from typing import Dict

from modules.hyperparameters.domain.dtos.training_hyperparameter_dto import TrainingHyperparameterDto
from modules.hyperparameters.domain.dtos.dataset_hyperparameter_dto import DatasetHyperparameterDto
from modules.hyperparameters.domain.dtos.output_hyperparameter_dto import OutputHyperparameterDto
from modules.hyperparameters.domain.dtos.layer_hyperparameter_dto import LayerHyperparameterDto
from modules.hyperparameters.domain.dtos.model_hyperparameter_dto import ModelHyperparameterDto

class HyperparametersDto(BaseModel):
    """
    DTO (Data Transfer Object) that encapsulates all the hyperparameters necessary 
    for configuring a machine learning model, including model configuration, 
    dataset configuration, layer configuration, training configuration, and output configuration.
    
    Attributes:
    - model (ModelHyperparameterDto): The configuration for the model hyperparameters, such as dataset and training information.
    - dataset (DatasetHyperparameterDto): The configuration for the dataset hyperparameters, including address and samples.
    - layers (Dict[str, LayerHyperparameterDto]): A dictionary of hyperparameters for the layers of the model, with layer names as keys.
    - training (TrainingHyperparameterDto): The configuration for the training hyperparameters, such as the learning rate and number of epochs.
    - output (OutputHyperparameterDto): The configuration for the output hyperparameters, such as model save path and checkpoint settings.
    
    Each of these parameters must be validated by their respective Pydantic DTOs.
    """
    
    model: ModelHyperparameterDto = Field(
        ...,
        description="The hyperparameters for the model, including the dataset, layers, training, and output configurations."
    )
    
    dataset: DatasetHyperparameterDto = Field(
        ...,
        description="The dataset hyperparameters, including the address of the dataset and samples to be used for training."
    )
    
    layers: Dict[str, LayerHyperparameterDto] = Field(
        ...,
        description="A dictionary of hyperparameters for each layer in the model, where the keys are the layer names and the values are the layer configurations (activation function, number of nodes, etc.)."
    )
    
    training: TrainingHyperparameterDto = Field(
        ...,
        description="The hyperparameters for training the model, such as the learning rate, batch size, and number of epochs."
    )
    
    output: OutputHyperparameterDto = Field(
        ...,
        description="The hyperparameters related to the output configuration, such as the path to save the model and checkpoint frequency."
    )

