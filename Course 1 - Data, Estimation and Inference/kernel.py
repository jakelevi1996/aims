import numpy as np

class SquaredExponential:
    def __init__(self, length_scale, kernel_scale):
        self._length_scale = length_scale
        self._kernel_scale = kernel_scale

    def __call__(self, x1, x2):
        sq_distance = np.square((x1 - x2) / self._length_scale)
        k = self._kernel_scale * np.exp(-sq_distance)
        return k

    def __repr__(self):
        s = (
            "SquaredExponential(length_scale=%r, kernel_scale=%r)"
            % (self._length_scale, self._kernel_scale)
        )
        return s

class Periodic:
    def __init__(self, period, length_scale, kernel_scale):
        self._angular_freq = 2 * np.pi / period
        self._length_scale = length_scale
        self._kernel_scale = kernel_scale

    def __call__(self, x1, x2):
        k = (
            self._kernel_scale
            * np.exp(
                np.cos(self._angular_freq * (x1 - x2))
                / self._length_scale
            )
        )
        return k

    def __repr__(self):
        period = 2 * np.pi / self._angular_freq
        s = (
            "Periodic(period=%r, length_scale=%r, kernel_scale=%r)"
            % (period, self._length_scale, self._kernel_scale)
        )
        return s

class Linear:
    def __init__(self, length_scale, centre):
        self._sq_length_scale = length_scale * length_scale
        self._centre = centre

    def __call__(self, x1, x2):
        k = (x1 - self._centre) * (x2 - self._centre) / self._sq_length_scale
        return k

    def __repr__(self):
        length_scale = np.sqrt(self._sq_length_scale)
        s = (
            "Linear(length_scale=%r, centre=%r)"
            % (self._centre, length_scale)
        )
        return s
