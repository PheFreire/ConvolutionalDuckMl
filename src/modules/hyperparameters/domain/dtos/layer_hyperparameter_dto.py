from typing import Literal

from pydantic import BaseModel, Field


class LayerHyperparameterDto(BaseModel):
    """
    DTO (Data Transfer Object) representing the hyperparameters of a layer in a neural network model.

    Attributes:
    - activation (str): The activation function used in the layer. Can be one of 'relu', 'sigmoid', 'softmax', etc.
    - num_nodes (int): The number of nodes (neurons) in this layer. Must be at least 1.
    """

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
        description="The activation function applied to the layer's output.",
        default="relu",
    )

    num_nodes: int = Field(
        description="The number of neurons (nodes) in this layer. Determines the layer's capacity.",
        default=3,
        ge=1,
    )
