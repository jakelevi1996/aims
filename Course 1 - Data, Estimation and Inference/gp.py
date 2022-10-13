import numpy as np

class GaussianProcess:
    def __init__(self, prior_mean_func, kernel_func, noise_std, rng=None):
        self._prior_mean_func = prior_mean_func
        self._kernel_func = kernel_func
        self._noise_var = noise_std * noise_std

        if rng is None:
            rng = np.random.default_rng()

        self._rng = rng
        self._conditioned = False

    def sample_prior(self, x):
        mean = self._prior_mean_func(x)
        cov = self._get_cov(x)
        root_cov = np.linalg.cholesky(cov)
        pre_transform_samples = self._rng.normal(size=x.shape)
        samples = root_cov @ pre_transform_samples + mean
        return samples

    def condition(self, x, y):
        self._x = x
        cov = self._get_cov(x)
        self._precision = np.linalg.inv(cov)
        self._scaled_error = self._precision @ (y - self._prior_mean_func(x))
        self._conditioned = True

    def predict(self, x_pred):
        if not self._conditioned:
            raise RuntimeError("Must condition on data before predicting")

        k_pred_data = self._kernel_func(
            x_pred.reshape(-1, 1, 1),
            self._x.reshape(1, 1, -1),
        )

        mean_prior = self._prior_mean_func(x_pred).reshape(-1, 1)
        mean_pred = mean_prior + k_pred_data @ self._scaled_error

        k_pred_pred = self._kernel_func(
            x_pred.reshape(-1, 1, 1),
            x_pred.reshape(-1, 1, 1),
        )
        k_data_pred = k_pred_data.transpose([0, 2, 1])

        var_pred = k_pred_pred - (k_pred_data @ self._precision @ k_data_pred)
        std_pred = np.sqrt(var_pred)

        return mean_pred.reshape(-1), std_pred.reshape(-1)

    def _get_cov(self, x):
        k_data_data = self._kernel_func(x.reshape(-1, 1), x.reshape(1, -1))
        cov = k_data_data + self._noise_var * np.identity(x.size)
        return cov
