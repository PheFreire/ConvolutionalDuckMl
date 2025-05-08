from config.app_error import AppError
import os

class Envs:
    def get(self, env: str) -> str:
        """
        Get env or raise error
        """
        value = os.getenv(env)
        if value is None:
            raise AppError(self, "Env Key Error", f"environment variable '{env}' not found!", code=500)

        return value

    @property
    def hyperparameters_path(self) -> str:
        """
        # Retrieves the path to the file containing hyperparameters, which is expected to be:

        - The **path to a TOML file** where the hyperparameters for the system or model are configured.

        # The environment variable `HYPERPARAMETERS_PATH` will store the path to the TOML file.
        
        # Returns:
            - str: The path to the hyperparameters file (typically a TOML file).

        # The variable `HYPERPARAMETERS_PATH` is expected to be defined in the system's environment variables file (`.env`|`.envrc`).
        """
        return self.get('HYPERPARAMETERS_PATH')

