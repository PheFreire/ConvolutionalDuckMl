from random import shuffle
from typing import List, TypeVar

T = TypeVar('T')

def get_random(data: List[T]) -> List[T]:
    shuffle(data)
    return data

