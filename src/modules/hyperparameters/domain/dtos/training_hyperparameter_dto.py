from typing import Literal

from pydantic import BaseModel, Field


class TrainingHyperparameterDto(BaseModel):
    """
    DTO (Data Transfer Object) representing the hyperparameters for training a machine learning model.

    Attributes:
    - gradient_descendent (str): The type of gradient descent to be used in the training process. Options include 'batch', 'stochastic', or 'mini_batch'.
    - learning_rate (float): The learning rate used for updating the weights during training.
    - batch_size (int): The number of samples to be processed in one forward/backward pass during training.
    - num_epochs (int): The total number of training epochs, or iterations over the entire dataset.
    """

    gradient_descendent: Literal["batch", "stochastic", "mini_batch"] = Field(
        description="Type of gradient descent to be used during training. Can be one of: 'batch', 'stochastic', or 'mini_batch'.",
        default="batch",
    )

    learning_rate: float = Field(
        description="Learning rate for updating model weights during training. Controls the step size during optimization.",
        default=0.001,
        ge=0.0,
    )

    batch_size: int = Field(
        description="Number of samples per batch during training. A larger batch size can speed up training but may require more memory.",
        default=100,
        ge=1,
    )

    num_epochs: int = Field(
        description="Total number of epochs for training. Defines how many times the model will iterate over the entire dataset.",
        default=100,
        ge=1,
    )
