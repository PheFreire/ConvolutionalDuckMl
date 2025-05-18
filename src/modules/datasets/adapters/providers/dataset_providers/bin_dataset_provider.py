from modules.datasets.domain.interfaces.providers import IDatasetProvider
from modules.hyperparameters.domain.dtos import SampleHyperparameterDto
from modules.neural_network.domain.interfaces.providers import ITensor
from modules.datasets.domain.dtos import DatasetDto, SampleDto
from framework.app_error import AppError
from io import BufferedReader
from typing import Iterable, Optional
from struct import unpack
import struct
import cv2

class BinDatasetProvider(IDatasetProvider):
    def __init__(
        self, 
        dataset_buffer: Optional[BufferedReader]=None, 
        sample_hyperparameter: Optional[SampleHyperparameterDto]=None,
        label: Optional[float]=None,
    ) -> None:
        self.__dataset_buffer = dataset_buffer
        self.__sample_hyperparameter = sample_hyperparameter
        self.__label = label
    
    @property
    def label(self) -> float:
        if isinstance(self.__label, float):
            return self.__label

        raise AppError(
            self, 
            'Dataset Label Type Error While Unpack', 
            '"label" must be "float"',
            code=500,
        )

    @property
    def dataset_buffer(self) -> BufferedReader:
        if isinstance(self.__dataset_buffer, BufferedReader):
            return self.__dataset_buffer

        raise AppError(
            self, 
            'Dataset Buffer Type Error While Unpack', 
            '"dataset_buffer" must be "BufferedReader"',
            code=500,
        )
    
    @property
    def sample_hyperparameter(self) -> SampleHyperparameterDto:
        if isinstance(self.__sample_hyperparameter, SampleHyperparameterDto):
            return self.__sample_hyperparameter
        
        raise AppError(
            self, 
            'Sample Hyperparameter Type Error While Unpack', 
            '"sample_hyperparameter" must be "SampleHyperparameterDto"',
            code=500,
        )

    def unpack(self, tensor: ITensor) -> DatasetDto:
        self.tensor = tensor
        
        return DatasetDto(lambda: self.unpacker(tensor), self.sample_hyperparameter.name)

    def __unpack_one(self, tensor: ITensor) -> Optional[SampleDto]:
        image = tensor.zeros(256, 256)

        _ = self.dataset_buffer.read(8)
        _ = self.dataset_buffer.read(2)
        _ = self.dataset_buffer.read(1)
        _ = self.dataset_buffer.read(4)

        n_strokes_data = self.dataset_buffer.read(2)

        if len(n_strokes_data) < 2:
            return
    
        n_strokes, = unpack('H', n_strokes_data)
        for _ in range(n_strokes):
            n_points_data = self.dataset_buffer.read(2)
            if len(n_points_data) < 2:
                return

            n_points, = unpack("H", n_points_data)

            x_data = self.dataset_buffer.read(n_points)
            y_data = self.dataset_buffer.read(n_points)
            if len(x_data) < n_points or len(y_data) < n_points:
                return

            x_vec = struct.unpack(f"{n_points}B", x_data)
            y_vec = struct.unpack(f"{n_points}B", y_data)


            for i in range(len(x_vec) - 1):
                pt1 = (int(x_vec[i]), int(y_vec[i]))
                pt2 = (int(x_vec[i + 1]), int(y_vec[i + 1]))
                cv2.line(image.value, pt1, pt2, color=(255,), thickness=1)
        
        return SampleDto(image / 255, tensor.new([self.label]))

    def unpacker(self, tensor: ITensor) -> Iterable[SampleDto]:
        quantity = self.sample_hyperparameter.quantity
        iterations = 0

        while True:
            if isinstance(quantity, int) and iterations >= quantity:
                break

            try:
                res = self.__unpack_one(tensor)
                if res is None:
                    break

                yield res

            except struct.error:
                break
