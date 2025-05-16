import numpy as np
from neural_network import NeuralNetwork
from numpy.typing import NDArray


class Training:
    def __init__(self, l_rate: float, network: NeuralNetwork) -> None:
        self.network = network
        self.l_rate = np.float64(l_rate)

    def train(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int,
        batch_size: int,
    ):
        n_samples = x.shape[0]

        for epoch in range(epochs):
            # funciona como o range(0, n_samples)
            indices = np.arange(n_samples)

            # Embaralha os elementos para usar dados em ordem aleatória
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]

            epoch_losses = []

            for i in range(0, n_samples, batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                batch_losses = []

                for x_i, y_i in zip(x_batch, y_batch):
                    loss = self.network.propagate(x_i, y_i)
                    self.network.backpropagate(self.l_rate)
                    batch_losses.append(loss)

                batch_mean_loss = np.mean(batch_losses)
                epoch_losses.extend(batch_losses)

                print(
                    f"[Epoch {epoch+1}] Batch {i//batch_size+1} Loss: {batch_mean_loss:.6f}"
                )

            epoch_mean_loss = np.mean(epoch_losses)
            print(
                f"➡️ Epoch {epoch+1}/{epochs} complete - Mean Loss: {epoch_mean_loss:.6f}\n"
            )
