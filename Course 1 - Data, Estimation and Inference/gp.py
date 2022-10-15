import numpy as np

LOG_2_PI = np.log(2 * np.pi)

class GaussianProcess:
    def __init__(self, prior_mean_func, kernel_func, noise_std, rng=None):
        self._prior_mean_func = prior_mean_func
        self._kernel_func = kernel_func
        self._noise_var = noise_std * noise_std
        self._rng = rng
        self._conditioned = False

    def sample_prior(self, x, n_samples=1):
        mean = self._prior_mean_func(x)
        cov = self._get_cov(x)
        root_cov = np.linalg.cholesky(cov)
        untransformed_samples = self._get_normal_samples([x.size, n_samples])
        samples = root_cov @ untransformed_samples + mean.reshape(-1, 1)
        return samples

    def condition(self, x, y):
        x = np.reshape(x, [-1])
        y = np.reshape(y, [-1])
        if not self._conditioned:
            self._x = x
            cov = self._get_cov(x)
            self._precision = np.linalg.inv(cov)
            self._error = y - self._prior_mean_func(x)
            self._scaled_error = self._precision @ self._error
            self._conditioned = True
        else:
            k_new = self._get_cov(x)
            k_old_new = self._kernel_func(
                self._x.reshape(-1, 1),
                x.reshape(1, -1),
            )
            scaled_k = self._precision @ k_old_new
            precision_new_new = np.linalg.inv(k_new - k_old_new.T @ scaled_k)
            precision_old_new = -scaled_k @ precision_new_new
            precision_old_old = (
                self._precision - (precision_old_new @ scaled_k.T)
            )
            self._precision = np.block(
                [
                    [precision_old_old, precision_old_new],
                    [precision_old_new.T, precision_new_new],
                ]
            )
            self._x = np.block([self._x, x])
            error_new = y - self._prior_mean_func(x)
            self._error = np.block([self._error, error_new])
            self._scaled_error = self._precision @ self._error

    def decondition(self):
        self._conditioned = False

    def predict(self, x_pred):
        self._assert_conditioned()

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

    def sample_posterior(self, x_pred, n_samples=1):
        self._assert_conditioned()

        k_pred_data = self._kernel_func(
            x_pred.reshape(-1, 1),
            self._x.reshape(1, -1),
        )

        mean_prior = self._prior_mean_func(x_pred)
        mean_pred = mean_prior + k_pred_data @ self._scaled_error

        k_pred_pred = self._get_cov(x_pred)
        k_data_pred = k_pred_data.T

        var_pred = k_pred_pred - (k_pred_data @ self._precision @ k_data_pred)

        root_var = np.linalg.cholesky(var_pred)
        samples_shape = [x_pred.size, n_samples]
        untransformed_samples = self._get_normal_samples(samples_shape)
        samples = root_var @ untransformed_samples + mean_pred.reshape(-1, 1)
        return samples

    def log_marginal_likelihood(self):
        self._assert_conditioned()

        sign, log_det_precision = np.linalg.slogdet(self._precision)
        if sign <= 0:
            raise RuntimeError("Determinant of precision is not positive!")

        m = -0.5 * (
            self._error @ self._scaled_error
            - log_det_precision
            + self._x.size * LOG_2_PI
        )
        return m

    def rmse(self, x, y):
        self._assert_conditioned()

        k_pred_data = self._kernel_func(
            x.reshape(-1, 1),
            self._x.reshape(1, -1),
        )

        mean_prior = self._prior_mean_func(x)
        mean_pred = mean_prior + k_pred_data @ self._scaled_error
        error = mean_pred - y

        return np.sqrt(np.mean(np.square(error)))

    def _get_cov(self, x):
        k_data_data = self._kernel_func(x.reshape(-1, 1), x.reshape(1, -1))
        cov = k_data_data + self._noise_var * np.identity(x.size)
        return cov

    def _get_normal_samples(self, shape):
        if self._rng is None:
            self._rng = np.random.default_rng()

        return self._rng.normal(size=shape)

    def _assert_conditioned(self):
        if not self._conditioned:
            raise RuntimeError(
                "Must condition on data before calling this method"
            )
