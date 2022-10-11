import numpy as np

class SquaredExponential:
    def __init__(self, length_scale, kernel_scale):
        self._length_scale = length_scale
        self._kernel_scale = kernel_scale

    def __call__(self, x1, x2):
        sq_distance = np.square((x1 - x2) / self._length_scale)
        k = self._kernel_scale * np.exp(-sq_distance)
        return k
