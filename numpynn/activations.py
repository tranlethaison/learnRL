import numpy as np


class Linear:
    @staticmethod
    def f(x):
        return x

    @staticmethod
    def de(x):
        """Return derivative."""
        return 1


class Sigmoid:
    @staticmethod
    def f(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def de(x):
        """Return derivative."""
        return Sigmoid.f(x) * (1.0 - Sigmoid.f(x))
