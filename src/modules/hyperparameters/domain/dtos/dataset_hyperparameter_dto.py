from pydantic import BaseModel, Field
from typing import List, Literal, Union


ADRESS_DOC = """
# Retrieves the address of the dataset, which can either be:

- A **database connection URL** for connecting to a remote or local database where dataset samples are stored.
- A **directory path** to a local directory where dataset samples are stored.

# The exact behavior depends on the dependency injection configuration. 

- If the system is set up to connect to a database, the environment variable `DATASET_ADDRESS` will store the connection URL.
- If the system is configured to use local files, the environment variable will contain the path to the dataset directory.
        
# Returns:
    - str: The dataset address (either a database connection URL or a directory path).

# The variable `DATASET_ADDRESS` is expected to be defined in the project hyperparameters file (`.toml`).
"""

class SampleHyperparameterDto(BaseModel):
    """
    # DTO for sample hyperparameters, representing a sample in the dataset.
    
    Attributes:
    - name (str): The name of the sample.
    - quantity (int | '*'): The quantity of this sample in the dataset.
    """
    name: str
    quantity: Union[int, Literal['*']]

class DatasetHyperparameterDto(BaseModel):
    """
    # DTO for dataset hyperparameters, including the address and samples used for training.
    
    Attributes:
    - address (str): The address of the dataset, which can be a database URL or a file path.
    - samples (List[SampleHyperparameterDto]): List of samples of the dataset used for model training.
    """
    address: str = Field(
        description=ADRESS_DOC,
        default='/home/dev/datasets/exemple_1.bin', 
        min_length=1,
    )

    samples: List[SampleHyperparameterDto] = Field(
        description="List of samples of the dataset used to train the model",
        default_factory=lambda: [SampleHyperparameterDto(name='default_sample_name', quantity=100)],
        min_length=1,
    )

