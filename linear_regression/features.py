import numpy as np

class _Features:
    def __call__(self, X):
        raise NotImplementedError()

class Linear(_Features):
    def __call__(self, X):
        return X

class Affine(_Features):
    def __call__(self, X):
        N, D = X.shape
        return np.block([X, np.ones([N, 1])])

class Polynomial(_Features):
    def __init__(self, degree):
        self._degree = degree

    def __call__(self, X):
        N, D = X.shape
        phi = np.ones([N, 1 + self._degree * D])
        p = 1
        X_pow = np.array(X)
        for _ in range(self._degree):
            phi[:, p:(p + D)] = X_pow
            X_pow *= X
            p += D

        return phi
