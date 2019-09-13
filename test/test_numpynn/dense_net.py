import sys
sys.path.append(".")

import numpy as np
from keras.datasets import mnist

from numpynn.layers import Dense
from numpynn.activations import Sigmoid
from numpynn.initializers import RandomNormal, RandomUniform, Zeros
from numpynn.models import Model
from numpynn.optimizers import SGD
from numpynn.losses import MSE


if __name__ == "__main__":
    # << Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    def preprocess(x):
        return np.expand_dims((x / 255.0).reshape(x.shape[0], -1), axis=-1)

    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    eye = np.eye(10)
    y_train = np.expand_dims(eye[y_train], axis=-1)
    # y_test = eye[y_test]
    # >> Data

    inputs = Dense(784)
    x = Dense(30, activation=Sigmoid)(inputs)
    outputs = Dense(10, activation=Sigmoid)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=SGD(lr=3.0), loss=MSE)
    model.fit(x_train, y_train, batch_size=10, epochs=30, test_data=(x_test, y_test))
