from typing import Any, Dict
from modules.hyperparameters.domain.interfaces.providers import IHyperParserProvider
from framework.parser.adapters.toml_parser import TomlParser
from framework.envs import Envs

class TomlHyperParserProvider(IHyperParserProvider):
    def __init__(self) -> None:
        self.env = Envs()
        self.parser = TomlParser()

    def parse(self) -> Dict[str, Any]:
        return self.parser.load(self.env.hyperparameters_path)
