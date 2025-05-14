import os
import framework
import numpy as np
from layer_setup import LayerSetup
from modules.hyperparameters.domain.orchestrators.load_hyperparameters_orchestrator import LoadHyperparametersOrchestrator
from neural_network import NeuralNetwork
from parser import unpack_drawings
from training import Training

load_hyperparameters_orchestrator = LoadHyperparametersOrchestrator()
repository = load_hyperparameters_orchestrator.execute()
print(repository)
