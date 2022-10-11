import numpy as np

class GaussianProcess:
    def __init__(self, prior_mean_func, kernel_func, noise_std, rng=None):
        self._prior_mean_func = prior_mean_func
        self._kernel_func = kernel_func
        self._noise_var = noise_std * noise_std

        if rng is None:
            rng = np.random.default_rng()

        self._rng = rng

    def sample_prior(self, x):
        mean = self._prior_mean_func(x)
        cov = self._get_cov(x)
        root_cov = np.linalg.cholesky(cov)
        pre_transform_samples = self._rng.normal(size=x.shape)
        samples = root_cov @ pre_transform_samples + mean
        return samples

    def _get_cov(self, x):
        k_data_data = self._kernel_func(x.reshape(-1, 1), x.reshape(1, -1))
        cov = k_data_data + self._noise_var * np.identity(x.size)
        return cov
