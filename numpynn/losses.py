import numpy as np


class MSE:
    """Mean Squared Error."""

    @staticmethod
    def f(y_true, y_predict):
        return 0.5 * np.sum(np.square(y_predict - y_true), axis=0)

    @staticmethod
    def dydy_true(y_true, y_predict):
        """Return partial derivative wrt `y_true` (output activations)."""
        return y_predict - y_true
