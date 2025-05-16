from abc import ABC, abstractmethod

from modules.hyperparameters.domain.interfaces.states.i_validate_hyper_sections_state import \
    IValidateHyperSectionsState


class IReadHyperFileState(ABC):
    @abstractmethod
    def call(self) -> IValidateHyperSectionsState:
        """
        # State responsible for reading the hyperparameter file.
            - This state is tasked with reading the hyperparameter file and transitioning to the next state, which is responsible for validating the model section of the hyperparameters.

        # Methods:
            - call: Initiates the transition to the next state, which validates the model section of the hyperparameters.

        # Parameters:
            - None

        # Returns:
            - IValidateModelSectionState: The next state responsible for validating the model section.
        """
        pass
