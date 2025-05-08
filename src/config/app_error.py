from typing import Any, Dict
import json

class AppError(Exception):
    """
    Custom exception class to represent application errors with detailed information.
    
    Attributes:
    - class_pointer: The class instance where the error originated.
    - title: A short, human-readable title for the error.
    - message: A detailed message describing the error.
    - details: Additional details for the error, typically a dictionary with further information.
    - code: An HTTP-like status code representing the type of error (default is 400).
    """
    
    def __init__(
        self, 
        class_pointer: Any,
        title: str,
        message: str = "", 
        details: Dict[str, Any] = {},
        code: int = 400,
    ) -> None:
        self.class_pointer = class_pointer
        self.title = title
        self.message = message
        self.details = details
        self.code = code
    
    @property
    def error(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_pointer.__class__.__name__,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "code": self.code,
        }

    def __str__(self) -> str:
        error_msg = f"({self.code})[{self.title.upper()}]: {self.message}\n\n"
        error_msg += json.dumps(self.error, ensure_ascii=False, indent=3)
        return error_msg
