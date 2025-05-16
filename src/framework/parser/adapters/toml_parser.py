from typing import Any, Dict

import toml

from framework.app_error import AppError
from framework.parser.interfaces.parser import IParser


class TomlParser(IParser):
    def load(self, path: str) -> Dict[str, Any]:
        """
        # Function that loads the TOML file from the given path and raises an AppError if something goes wrong.

        # Args:
            - path (str): The path to the TOML file to load.

        # Returns:
            - dict: Parsed content of the TOML file.

        # Raises:
            - AppError: If the TOML file cannot be loaded, raises an AppError with details.
        """
        try:
            return toml.load(path)
        except Exception as e:
            raise AppError(
                class_pointer=self,
                title="TomlFileLoadError",
                message="Failed to load the TOML file.",
                details={"exception_details": str(e), "invalid_path": path},
                code=500,
            )
