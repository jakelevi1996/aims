import numpy as np

class SquaredExponential:
    def __init__(self, length_scale, kernel_scale):
        self._length_scale = length_scale
        self._kernel_scale = kernel_scale

    def get_parameter_vector(self):
        return np.log([self._length_scale, self._kernel_scale])

    def set_parameter_vector(self, param_vector):
        [self._length_scale, self._kernel_scale] = np.exp(param_vector)

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

    def get_parameter_vector(self):
        param_list = [
            self._angular_freq,
            self._length_scale,
            self._kernel_scale,
        ]
        return np.log(param_list)

    def set_parameter_vector(self, param_vector):
        [
            self._angular_freq,
            self._length_scale,
            self._kernel_scale,
        ] = np.exp(param_vector)

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

    def get_parameter_vector(self):
        return [np.log(self._sq_length_scale), self._centre]

    def set_parameter_vector(self, param_vector):
        log_sq_length_scale, self._centre = param_vector
        self._sq_length_scale = np.exp(log_sq_length_scale)

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

class _Reduction:
    def __init__(self, *kernels):
        self._kernels = kernels

    def get_parameter_vector(self):
        parameter_vector = np.concatenate(
            [k.get_parameter_vector() for k in self._kernels]
        )
        return parameter_vector

    def set_parameter_vector(self, param_vector):
        num_param_list = [
            len(k.get_parameter_vector()) for k in self._kernels
        ]
        split_inds = np.cumsum(num_param_list)
        param_list = np.split(param_vector, split_inds)
        for kernel, params in zip(self._kernels, param_list):
            kernel.set_parameter_vector(params)

class Sum(_Reduction):
    def __call__(self, x1, x2):
        return sum(k(x1, x2) for k in self._kernels)

    def __repr__(self):
        return "Sum(%s)" % (", ".join(repr(k) for k in self._kernels))

class Product(_Reduction):
    def __call__(self, x1, x2):
        k = self._kernels[0](x1, x2)
        for kernel in self._kernels[1:]:
            k *= kernel(x1, x2)
        return k

    def __repr__(self):
        return "Product(%s)" % (", ".join(repr(k) for k in self._kernels))
