import numpy as np

class ZeroMean:
    def __call__(self, x):
        return np.zeros(x.shape)

class Constant:
    def __init__(self, offset):
        self._offset = offset

    def __call__(self, x):
        return np.full(x.shape, self._offset)

class Linear:
    def __init__(self, scale, offset):
        self._scale = scale
        self._offset = offset

    def __call__(self, x):
        return (self._scale * x) + self._offset
