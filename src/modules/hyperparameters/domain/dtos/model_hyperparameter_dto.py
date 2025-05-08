from pydantic import BaseModel, Field
from typing import List

class ModelHyperparameterDto(BaseModel):
    """
    DTO (Data Transfer Object) representing the hyperparameters for configuring the model.
    
    Attributes:
    - plugged_dataset (str): The dataset to be used for training. Specifies the name of the dataset configuration template.
    - plugged_layers (List[str]): A list of layer configuration templates used for defining the hidden layers in the model.
    - plugged_training (str): The training configuration template used for training the model.
    - plugged_output (str): The output configuration template, defining how the model's output is managed.
    """
    
    plugged_dataset: str = Field(
        description="The dataset used for training the model. Specifies the name of the dataset configuration.",
        default='default_dataset', 
        min_length=1,
    )

    plugged_layers: List[str] = Field(
        description="A list of the hidden layer configuration templates used in the model.",
        default=['default_hidden_1'],
        min_length=1,
    )

    plugged_training: str = Field(
        description="The name of the neural network training configuration template used for the model's training process.",
        default='default_training',
        min_length=1,
    )

    plugged_output: str = Field(
        description="The name of the neural network output configuration template, specifying how the model's output is handled.",
        default='default_output',
    )

