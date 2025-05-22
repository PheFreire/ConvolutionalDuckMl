from typing import Any
from framework.app_error import AppError

def get_field[T](cls: Any, struct: dict[str, T], key: str) -> T:
    data = struct.get(key)
    if data is not None:
        return data

    raise AppError(
        class_pointer=cls,
        title="Key Error",
        message=f"Key '{key}' not found!",
        details={"missing_key": key, "struct": struct},
        code=500,
    )
