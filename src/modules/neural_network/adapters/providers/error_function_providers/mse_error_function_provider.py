from modules.neural_network.domain.interfaces.providers import ITensor, IErrorFunctionProvider
from framework.app_error import AppError
from typing import Optional

class MseErrorFunctionProvider(IErrorFunctionProvider):
    def __init__(self) -> None:
        self.last_distance: Optional[ITensor] = None

    def distance(self, predicted_y: ITensor, true_y: ITensor) -> ITensor:
        tensor_type = type(predicted_y)

        distance = true_y - predicted_y
        self.last_distance = distance

        distance_squared_sum = (distance ** 2).sum()
        element_count = distance.shape()[0]

        return tensor_type.new([distance_squared_sum / element_count])

    def backpropagate(self) -> ITensor:
        if self.last_distance is None:
            raise AppError(
                self, 
                "Error Backpropagating Error Function", 
                "Call distance() before execute backpropagate()."
            )
        return self.last_distance

