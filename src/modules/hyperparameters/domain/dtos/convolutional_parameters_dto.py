from typing import List
from pydantic import BaseModel, Field

class ConvolutionalParametersDto(BaseModel):
    """
    DTO containing the parameters required to configure a convolutional layer.

    Attributes:
    - in_channels (int): Number of input channels to the layer (e.g., 3 for RGB images).
    - kernel (List[int]): Size of the convolutional kernel. Typically a 2-element list [height, width].
    - stride (int): Number of pixels to move the kernel each step. Default is 1.
    - padding (int): Number of pixels to pad around the input. Default is 0.
    """

    in_channels: int = Field(
        description="Number of input channels. For example, 3 for RGB images or the number of filters from the previous convolutional layer.",
        ge=1,
    )

    kernel: List[int] = Field(
        description="Dimensions of the kernel as a list [height, width]. Each value must be >= 1.",
        min_length=2,
    )

    stride: int = Field(
        description="Stride of the convolution. Indicates how far the kernel moves after each application.",
        default=1,
        ge=1,
    )

    padding: int = Field(
        description="Padding applied to all sides of the input. Zero-padding extends the borders with zeros.",
        default=0,
        ge=0,
    )

