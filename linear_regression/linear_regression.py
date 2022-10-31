import numpy as np
import linear_regression.features

class LinearRegression:
    def __init__(self, features=None):
        self._params = None
        if features is None:
            features = linear_regression.features.Linear()
        self._features = features

    def estimate_ml(self, X, y, jitter=1e-8):
        """
        X: N x D matrix of training inputs
        y: N x 1 vector of training targets/observations
        Calculates maximum likelihood parameters (D x 1) and stores in _params
        """
        phi = self._features(np.array(X, float))
        N, D = phi.shape
        diag_inds = np.arange(D)
        phi_gram = phi.T @ phi
        phi_gram[diag_inds, diag_inds] += jitter

        self._params = np.linalg.solve(phi_gram, phi.T @ y)

    def estimate_map(self, X, y, sigma, alpha):
        """
        Phi: training inputs, Size of N x D
        y: training targets, Size of D x 1
        sigma: standard deviation of the noise
        alpha: standard deviation of the prior on the parameters
        Calculates MAP estimate of parameters (D x 1) and stores in _params
        """
        phi = self._features(np.array(X, float))
        N, D = phi.shape
        diag_inds = np.arange(D)
        phi_gram = phi.T @ phi
        phi_gram[diag_inds, diag_inds] += np.square(sigma / alpha)

        self._params = np.linalg.solve(phi_gram, phi.T @ y)

    def predict(self, X):
        """
        Xtest: K x D matrix of test inputs
        returns: prediction of f(Xtest); K x 1 vector
        """
        if self._params is None:
            raise ValueError("Must estimate parameters before predicting")

        return self._features(X) @ self._params

    def __repr__(self):
        s = (
            "LinearRegression(features=%s, params.T=%s)"
            % (self._features, self._params.T)
        )
        return s
