import numpy as np
from tqdm import tqdm


class SGD:
    """Stochatic Gradient Descent"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def optimize(self, model):
        """1 epoch optimization."""
        p_batches = tqdm(model.batches)
        for (x_train, y_train) in p_batches:
            x, y = x_train.T, y_train.T

            # Feedforward
            z = [None] * len(model.layers)
            a = [None] * len(model.layers)

            a[0] = x
            for l in range(1, len(model.layers)):
                z[l] = (
                    np.matmul(model.layers[l].weights, a[l - 1]) + model.layers[l].bias
                )
                a[l] = model.layers[l].activation.f(z[l])

            # Ouput error
            delta = [None] * len(model.layers)

            delta[-1] = (
                model.loss.de_y_true(y, a[-1]) * model.layers[-1].activation.de(z[-1])
            )

            # Backpropagate
            for l in range(len(model.layers) - 2, 0, -1):
                delta[l] = (
                    np.matmul(model.layers[l + 1].weights.T, delta[l + 1])
                    * model.layers[l].activation.de(z[l])
                )

            # Gradient Descent
            m = x.shape[-1]
            for l in range(1, len(model.layers)):
                model.layers[l].weights -= self.lr / m * np.matmul(delta[l], a[l -1].T)
                model.layers[l].bias -= self.lr * np.mean(delta[l], axis=-1, keepdims=1)

            p_batches.set_description("Batches")

        return 0
