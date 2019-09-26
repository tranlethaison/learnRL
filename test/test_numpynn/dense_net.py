import sys

sys.path.append(".")

import numpy as np
from keras.datasets import mnist, fashion_mnist

from numpynn.layers import Dense
from numpynn.activations import Linear, Sigmoid
from numpynn.initializers import RandomNormal, RandomUniform, Zeros, StandardNormal
from numpynn.models import Model
from numpynn.optimizers import SGD
from numpynn.losses import MSE, CrossEntropy


def model_0():
    """Return MSE model."""
    inputs = Dense(784)
    x = Dense(
        30,
        activation=Sigmoid,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
    )(inputs)
    outputs = Dense(
        10,
        activation=Sigmoid,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=SGD(lr=3.0), loss=MSE, n_classes=10)
    return model


def model_1():
    """Return Cross-Entropy model."""
    inputs = Dense(784)
    x = Dense(
        30,
        activation=Sigmoid,
        kernel_initializer=StandardNormal(),
        bias_initializer=Zeros(),
    )(inputs)
    outputs = Dense(
        10,
        activation=Sigmoid,
        kernel_initializer=StandardNormal(),
        bias_initializer=Zeros(),
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=SGD(lr=0.5), loss=CrossEntropy, n_classes=10)
    return model


if __name__ == "__main__":
    # << Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    n = 50000
    x_train, x_val = x_train[:n], x_train[n:]
    y_train, y_val = y_train[:n], y_train[n:]

    def preprocess(x):
        return (x / 255.0).reshape(x.shape[0], -1)

    x_train = preprocess(x_train)
    x_val = preprocess(x_val)
    x_test = preprocess(x_test)
    # >> Data

    #model = model_0()
    model = model_1()

    model.fit(x_train, y_train, batch_size=10, n_epochs=30, val_data=(x_val, y_val))
    accuracy = model.evaluate(x_test, y_test) / len(x_test)
    print("Accuracy:", accuracy)
