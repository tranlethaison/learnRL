import numpy as np


class Linear:
    @staticmethod
    def f(z):
        return z

    @staticmethod
    def df(z):
        """Return derivative."""
        return 1


class Sigmoid:
    @staticmethod
    def f(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def df(z):
        """Return derivative."""
        return Sigmoid.f(z) * (1.0 - Sigmoid.f(z))
