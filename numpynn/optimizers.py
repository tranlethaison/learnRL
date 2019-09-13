import numpy as np
from tqdm import tqdm


class SGD:
    """Stochatic Gradient Descent"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def optimize(self, model):
        batch_losses = [None] * len(model.batches)

        p_batches = tqdm(model.batches)
        for bid, (x, y) in enumerate(p_batches):
            layers_samples_errors = [None]
            layers_samples_errors += [
                np.zeros((len(x), model.layers[l].units, 1), dtype=np.float64)
                for l in range(1, len(model.layers))
            ] 

            layers_samples_activations = [  
                np.zeros((len(x), model.layers[l].units, 1), dtype=np.float64)
                for l in range(len(model.layers) - 1)
            ]
            layers_samples_activations += [None]

            losses = [None] * len(x)
            
            for eid, (xx, yy) in enumerate(zip(x, y)):
                # Feedforward
                affines, activations = model.forward(xx)

                losses[eid] = model.loss.f(yy, activations[-1])

                # Output error
                layers_samples_errors[-1][eid] = (
                    model.loss.dydy_true(yy, activations[-1])
                    * model.layers[-1].activation.dydx(affines[-1])
                )

                # Backpropagate the error
                for l in range(len(model.layers) - 2, 0, -1):
                    layers_samples_errors[l][eid] = (
                        np.matmul(
                            model.layers[l + 1].weights.T, 
                            layers_samples_errors[l + 1][eid]
                        )
                        * model.layers[l].activation.dydx(affines[l])
                    )

                    layers_samples_activations[l][eid] = activations[l]
                layers_samples_activations[0][eid] = activations[0]

            # Gradient descent
            for l in range(len(model.layers) - 1, 0, -1):
                model.layers[l].bias -= self.lr * layers_samples_errors[l].mean(axis=0)
                model.layers[l].weights -= (
                    self.lr 
                    * np.matmul(
                        layers_samples_errors[l],
                        np.transpose(layers_samples_activations[l -1], axes=(0, 2, 1))
                    ).mean(axis=0)
                )
            
            batch_losses[bid] = np.mean(losses)

            p_batches.set_description("Batches")

        return batch_losses
