from graphviz import Digraph

from framework.envs import Envs
from modules.neural_network.domain.interfaces.providers import \
    INeuralNetworkProvider


def network_ui(network: INeuralNetworkProvider) -> str:
    """
    Generate a Graphviz-based diagram of a neural network's structure with labeled neurons
    showing their activation function, and a final loss node.

    Args:
        network (INeuralNetworkProvider): The network to visualize.

    Returns:
        str: The path to the rendered image file.
    """
    model_img_path = Envs().model_img_path
    g = Digraph(name="NeuralNetworkUI", format="png")
    g.attr(rankdir="LR", nodesep="0.6", splines="line")
    g.attr(label="Neural Network Structure", labelloc="t", fontsize="20")

    # Infer input size from the first perceptron's weights
    first_layer = network.layers[0]
    input_size = first_layer.perceptrons[0].w.shape()[0]

    # Input layer
    for i in range(input_size):
        g.node(f"input_{i}", f"x[{i}]", shape="box")

    # Hidden/output layers
    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.perceptrons):
            node_id = f"L{layer_index}_N{neuron_index}"
            label = f"Neuron {neuron_index}\n({neuron.activation_function.activation})"
            g.node(
                node_id, label, shape="circle", style="filled", fillcolor="lightblue"
            )

            if layer_index == 0:
                for i in range(input_size):
                    g.edge(f"input_{i}", node_id)
            else:
                for prev_index in range(
                    len(network.layers[layer_index - 1].perceptrons)
                ):
                    g.edge(f"L{layer_index - 1}_N{prev_index}", node_id)

    # Output neurons
    output_layer_index = len(network.layers) - 1
    for i in range(len(network.layers[-1].perceptrons)):
        g.node(f"output_{i}", f"y[{i}]", shape="doublecircle", color="green")
        g.edge(f"L{output_layer_index}_N{i}", f"output_{i}")

    # Loss node
    g.node("Loss", "Loss", shape="diamond", style="filled", fillcolor="pink")
    for i in range(len(network.layers[-1].perceptrons)):
        g.edge(f"output_{i}", "Loss", style="dashed", color="gray")

    return g.render("neural_network_ui", directory=model_img_path, view=False)
