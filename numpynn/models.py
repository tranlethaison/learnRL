from collections import deque

import numpy as np


class Model:
    """The Neural Net."""

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

        layers = deque()
        x = self.outputs
        while 1:
            if not hasattr(x, "inputs"):
                break
            x.init_bias()
            x.init_weights()

            layers.appendleft(x)
            x = x.inputs
        layers.appendleft(self.inputs)
        self.layers = list(layers)

    def fit(self, x, y, batch_size=None, epochs=1, test_data=None):
        assert len(x) == len(y)

        if not batch_size:
            batch_size = len(x)

        self.batches = [
            (x[i : i + batch_size], y[i : i + batch_size])
            for i in range(0, len(x), batch_size)
        ]

        for e in range(epochs):
            print("Epoch {}:".format(e))
            losses = self.optimizer.optimize(self)
            print("Loss: {}".format(np.mean(losses)))

            if test_data:
                print(
                    "Test result: {}/{}".format(
                        self.evaluate(*test_data), len(test_data[0])
                    )
                )
            print()

    def forward(self, a):
        affines = [None] * len(self.layers)
        activations = [None] * len(self.layers)

        activations[0] = a
        for l in range(1, len(self.layers)):
            affines[l] = (
                np.matmul(self.layers[l].weights, activations[l - 1])
                + self.layers[l].bias
            )
            activations[l] = self.layers[l].activation.f(affines[l])

        return affines, activations

    def predict(self, a):
        for l in range(1, len(self.layers)):
            a = self.layers[l].activation.f(
                np.matmul(self.layers[l].weights, a) + self.layers[l].bias
            )
        return a

    def evaluate(self, x, y):
        predicts = [self.predict(xx).argmax() for xx in x]
        return sum(int(p == yy) for p, yy in zip(predicts, y))
