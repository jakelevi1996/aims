import numpy as np

class GaussianProcess:
    def __init__(self, prior_mean_func, kernel_func, rng=None):
        self._prior_mean_func = prior_mean_func
        self._kernel_func = kernel_func

        if rng is None:
            rng = np.random.default_rng()

        self._rng = rng

    def sample_prior(self, x):
        mean = self._prior_mean_func(x)
        cov = self._kernel_func(x, x)
        root_cov = np.linalg.cholesky(cov)
        noise = np.random.normal(size=x.shape)
        samples = root_cov @ noise + mean
        return samples
