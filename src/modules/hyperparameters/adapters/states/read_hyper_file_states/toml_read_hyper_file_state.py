from framework.envs import Envs
from framework.parser import IParser
from modules.hyperparameters.adapters.states.validate_hyper_sections_states import \
    PydanticValidateHyperSectionsState
from modules.hyperparameters.domain.interfaces.states import (
    IReadHyperFileState, IValidateHyperSectionsState)


class TomlReadHyperFileState(IReadHyperFileState):
    def __init__(self, parser: IParser, env: Envs) -> None:
        self.path = env.hyperparameters_path
        self.parser = parser

    def call(self) -> IValidateHyperSectionsState:
        return PydanticValidateHyperSectionsState(self.parser.load(self.path))
