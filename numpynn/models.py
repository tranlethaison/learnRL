from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from .visualizer import Visualizer


class Model:
    """The Neural Net."""

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def compile(self, optimizer, loss, n_classes):
        self.optimizer = optimizer
        self.loss = loss
        self.n_classes = n_classes

        layers = deque()
        x = self.outputs
        while 1:
            if not hasattr(x, "prior_layer"):
                break
            x.init_bias()
            x.init_weights()

            layers.appendleft(x)
            x = x.prior_layer
        layers.appendleft(self.inputs)
        self.layers = list(layers)

    def fit(self, x, y, batch_size=None, n_epochs=1, val_data=None):
        assert len(x) == len(y)

        vis = Visualizer(n_epochs)

        y = np.eye(self.n_classes)[y]

        if not batch_size:
            batch_size = len(x)

        for e in range(n_epochs):
            print("Epoch {}:".format(e))

            loss = np.mean(
                [
                    np.mean(sample_losses)
                    for sample_losses in self.optimizer.optimize(self, x, y, batch_size)
                ]
            )
            print("Loss: {}".format(loss))
            vis.update_loss(loss)

            if val_data:
                accuracy = self.evaluate(*val_data) / len(val_data[0])
                print("Validation accuracy: {}".format(accuracy))
                vis.update_accu(accuracy)
            print()

            plt.pause(0.05)
        plt.show()

    def predict(self, a):
        a = np.expand_dims(a, axis=-1)
        for l in range(1, len(self.layers)):
            a = self.layers[l].activation.f(
                np.matmul(self.layers[l].weights, a) + self.layers[l].bias
            )
        return a.argmax()

    def evaluate(self, x, y):
        predicts = [self.predict(xx) for xx in x]
        accuracy = sum(int(p == yy) for p, yy in zip(predicts, y))
        return accuracy
