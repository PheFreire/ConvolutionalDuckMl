from pydantic import BaseModel, Field
from typing import Literal, Optional

from modules.hyperparameters.domain.dtos.convolutional_parameters_dto import ConvolutionalParametersDto


class LayerHyperparameterDto(BaseModel):
    """
    Data Transfer Object (DTO) representing the hyperparameters for a layer in a neural network model.

    Attributes:
    - propagation_type (str): Type of layer propagation. "convolutional" for Conv2D layers, "dense" for fully connected layers.
    - activation (str): Activation function applied after the layer's linear transformation.
    - nodes (int): Number of perceptrons (neurons or filters) in the layer. For convolutional layers, this represents the number of filters.
    - convolutional_parameters (ConvolutionalParametersDto, optional): Parameters specific to convolutional layers. Must be provided when propagation_type is "convolutional".
    """

    propagation_type: Literal["convolutional", "dense"] = Field(
        description='Type of the layer. Use "convolutional" for Conv2D layers or "dense" for fully connected layers.'
    )

    activation: Literal[
        "relu",
        "sigmoid",
        "softmax",
        "step",
        "leaky_relu",
        "tanh",
        "silu",
        "gelu",
    ] = Field(
        description="Activation function applied to the output of the layer.",
        default="relu",
    )

    nodes: int = Field(
        description="Number of nodes (neurons or filters) in the layer. For convolutional layers, this is the number of filters.",
        default=3,
        ge=1,
    )

    convolutional_parameters: Optional[ConvolutionalParametersDto] = Field(
        default=None,
        description="Parameters specific to convolutional layers. Required when propagation_type is 'convolutional'.",
    )

