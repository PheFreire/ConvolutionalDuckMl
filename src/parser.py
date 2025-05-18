import struct
from typing import Any, Dict, Optional
from struct import unpack
import numpy as np
import cv2

class Unpack:
    def __init__(self, filename: str) -> None:
        self.file_handle = open(filename, "rb")
        self.file_handle.seek(0)
    
    def unpack_drawing(self, label: int, image_size=256) -> Optional[Dict[str, Any]]:
        image = np.zeros((image_size, image_size), dtype=np.float64)

        _ = self.file_handle.read(8)
        _ = self.file_handle.read(2)
        _ = self.file_handle.read(1)
        _ = self.file_handle.read(4)

        n_strokes_data = self.file_handle.read(2)

        if len(n_strokes_data) < 2:
            return 
    
        n_strokes, = unpack('H', n_strokes_data)
        for _ in range(n_strokes):
            n_points_data = self.file_handle.read(2)
            if len(n_points_data) < 2:
                return

            n_points, = unpack("H", n_points_data)

            x_data = self.file_handle.read(n_points)
            y_data = self.file_handle.read(n_points)
            if len(x_data) < n_points or len(y_data) < n_points:
                return

            x_vec = struct.unpack(f"{n_points}B", x_data)
            y_vec = struct.unpack(f"{n_points}B", y_data)

            for i in range(len(x_vec) - 1):
                pt1 = (int(x_vec[i]), int(y_vec[i]))
                pt2 = (int(x_vec[i + 1]), int(y_vec[i + 1]))
                cv2.line(image, pt1, pt2, color=(255,), thickness=1)
        
        return { "y": label, "x": image / 255.0 }

    def __del__(self):
        if not self.file_handle.closed:
            self.file_handle.close()
    
dataset = Unpack('/home/lemon/Documentos/dev/Lemon/MlImage/dataset/sock.bin')

abc = []
while True:
    try:
        res = dataset.unpack_drawing(1)
        if res is None:
            break

        abc.append(res)
    except struct.error:
        break

print(len(abc))

