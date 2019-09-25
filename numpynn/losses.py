import numpy as np


class MSE:
    @staticmethod
    def f(y, a):
        return 0.5 * np.sum(np.square(y - a), axis=0)

    @staticmethod
    def df_da(y, a):
        """Return partial derivative wrt `a` (element-wise)."""
        return a - y


class CrossEntropy:
    @staticmethod
    def f(y, a):
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a), axis=0)

    @staticmethod
    def df_da(y, a):
        """Return partial derivative wrt `a` (element-wise)."""
        return -(y / a + (y - 1) / (1 - a))
