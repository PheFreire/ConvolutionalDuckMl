from pydantic import BaseModel, Field


class OutputHyperparameterDto(BaseModel):
    """
    DTO (Data Transfer Object) representing the hyperparameters for the output configuration of the model.

    Attributes:
    - path (str): The path where the model's output will be saved after training.
    - num_epoch_to_checkpoint (int): The number of epochs after which to save a checkpoint.
    - checkpoint (bool): Whether to save a checkpoint every 'num_epoch_to_checkpoint' epochs or only at the end of the training.
    - read_output_as_starter_checkpoint (bool): Whether to use the saved output as a starter checkpoint for further training.
    """

    path: str = Field(
        description="Path where the model output will be saved. It defines the location of the saved model after training.",
        default="/home/dev/outputs/output_name.h5",
        min_length=1,
    )

    num_epoch_to_checkpoint: int = Field(
        description="The number of epochs after which to save a checkpoint. Helps in restoring training in case of interruptions.",
        default=20,
        ge=1,
    )

    checkpoint: bool = Field(
        description="Whether to save a checkpoint every 'num_epoch_to_checkpoint' epochs or only at the end of training.",
        default=True,
    )

    read_output_as_starter_checkpoint: bool = Field(
        description="Whether to use the saved output as a starter checkpoint for further training. This allows resuming training from where it left off.",
        default=False,
    )
