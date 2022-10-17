import numpy as np

class ZeroMean:
    def get_parameter_vector(self):
        return []

    def set_parameter_vector(self, param_vector):
        if len(param_vector) > 0:
            raise ValueError(
                "Received %i params, expected 0"
                % len(param_vector)
            )

    def __call__(self, x):
        return np.zeros(x.shape)

    def __repr__(self):
        return "ZeroMean()"

class Constant:
    def __init__(self, offset):
        self._offset = offset

    def get_parameter_vector(self):
        return [self._offset]

    def set_parameter_vector(self, param_vector):
        [self._offset] = param_vector

    def __call__(self, x):
        return np.full(x.shape, self._offset)

    def __repr__(self):
        return "Constant(offset=%r)" % self._offset

class Linear:
    def __init__(self, scale, offset):
        self._scale = scale
        self._offset = offset

    def get_parameter_vector(self):
        return [self._scale, self._offset]

    def set_parameter_vector(self, param_vector):
        [self._scale, self._offset] = param_vector

    def __call__(self, x):
        return (self._scale * x) + self._offset

    def __repr__(self):
        return "Linear(scale=%r, offset=%r)" % (self._scale, self._offset)
