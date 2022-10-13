import numpy as np

class SquaredExponential:
    def __init__(self, length_scale, kernel_scale):
        self._length_scale = length_scale
        self._kernel_scale = kernel_scale

    def __call__(self, x1, x2):
        sq_distance = np.square((x1 - x2) / self._length_scale)
        k = self._kernel_scale * np.exp(-sq_distance)
        return k

class Linear:
    def __init__(self, length_scale, centre):
        self._sq_length_scale = length_scale * length_scale
        self._centre = centre

    def __call__(self, x1, x2):
        k = (x1 - self._centre) * (x2 - self._centre) / self._sq_length_scale
        return k
