from modules.hyperparameters.adapters.states.read_hyper_file_states import TomlReadHyperFileState
from modules.hyperparameters.domain.interfaces.factories import ILoadHyperFactory
from modules.hyperparameters.domain.interfaces.states import IReadHyperFileState
from config.parser import TomlParser
from config.envs import Envs

class TomlLoadHyperFactory(ILoadHyperFactory):
    def call(self) -> IReadHyperFileState:
        return TomlReadHyperFileState(TomlParser(), Envs())

