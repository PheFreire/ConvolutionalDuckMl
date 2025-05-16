from framework.envs import Envs
from framework.parser import TomlParser
from modules.hyperparameters.adapters.states.read_hyper_file_states import \
    TomlReadHyperFileState
from modules.hyperparameters.domain.interfaces.factories import \
    ILoadHyperFactory
from modules.hyperparameters.domain.interfaces.states import \
    IReadHyperFileState


class TomlLoadHyperFactory(ILoadHyperFactory):
    def call(self) -> IReadHyperFileState:
        return TomlReadHyperFileState(TomlParser(), Envs())
