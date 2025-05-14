from numpy.typing import NDArray
from typing import Optional
import numpy as np

from framework.app_error import AppError

class ErrorFunction:
    def __init__(self) -> None:
        self.last_distance: Optional[NDArray[np.float64]] = None

    def distance(self, predicted_y: NDArray[np.float64], true_y: NDArray[np.float64]) -> np.float64:
        distance = true_y - predicted_y
        self.last_distance = distance
        
        return np.sum(distance ** 2)/len(distance)
        
    def backpropagate(self) -> NDArray[np.float64]:
        if self.last_distance is None:
            raise AppError(
                self, 
                "Error Backpropagating Error Function", 
                "Call distance() before execute backpropagate().",
            )
        return self.last_distance


