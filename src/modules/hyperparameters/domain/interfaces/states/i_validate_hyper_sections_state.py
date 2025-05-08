from modules.hyperparameters.domain.interfaces.states.i_build_hyper_state import IBuildHyperState
from abc import ABC, abstractmethod

class IValidateHyperSectionsState(ABC):
    @abstractmethod
    def call(self) -> IBuildHyperState:
        """
        Consolidated State responsible for validating all sections of the hyperparameters:
        - Model Section
        - Layers Section
        - Dataset Section
        - Training Section
        - Output Section

        This state initiates the validation process for each of the hyperparameter sections in the correct order:
        1. Model
        2. Layers
        3. Dataset
        4. Training
        5. Output

        Methods:
            - call: Initiates the entire validation process and transitions between sections as needed.

        Parameters:
            None

        Returns:
            IValidateOutputSectionState: The final state responsible for validating the output section of the hyperparameters.
        """
        pass

