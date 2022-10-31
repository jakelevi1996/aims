import numpy as np
import __init__
import plotting

class LinearRegression:
    def __init__(self, features=None):
        self._params = None
        if features is None:
            features = Linear()
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

    def predict(self, X):
        """
        Xtest: K x D matrix of test inputs
        returns: prediction of f(Xtest); K x 1 vector
        """
        if self._params is None:
            raise ValueError("Must estimate parameters before predicting")

        return self._features(X) @ self._params

class _Features:
    def __call__(self, X):
        raise NotImplementedError()

class Linear(_Features):
    def __call__(self, X):
        return X

class Affine(_Features):
    def __call__(self, X):
        N, D = X.shape
        return np.block([X, np.ones([N, 1])])

class Polynomial(_Features):
    def __init__(self, degree):
        self._degree = degree

    def __call__(self, X):
        N, D = X.shape
        phi = np.ones([N, 1 + self._degree * D])
        p = 1
        X_pow = np.array(X)
        for _ in range(self._degree):
            phi[:, p:(p + D)] = X_pow
            X_pow *= X
            p += D

        return phi

# Define training set
X = np.array([-3, -1, 0, 1, 3]).reshape(-1,1) # 5x1 vector, N=5, D=1
y = np.array([-1.2, -0.7, 0.14, 0.67, 1.67]).reshape(-1,1) # 5x1 vector

# Plot the training set
data_line = plotting.Line(
    X,
    y,
    marker="+",
    ms=10,
    ls="",
    c="b",
    label="Data",
    zorder=30,
)
plotting.plot(
    data_line,
    axis_properties=plotting.AxisProperties("$x$", "$y$"),
    plot_name="Linear regression dataset",
)

# define a test set
Xtest = np.linspace(-5,5,100).reshape(-1,1) # 100 x 1 vector of test inputs

# predict the function values at the test points using the maximum likelihood
# estimator
model = LinearRegression()
model.estimate_ml(X, y)
mle_prediction = model.predict(Xtest)

# plot
mle_prediction_line = plotting.Line(
    Xtest,
    mle_prediction,
    c="r",
    label="MLE estimate",
    zorder=20,
)
plotting.plot(
    data_line,
    mle_prediction_line,
    axis_properties=plotting.AxisProperties("$x$", "$y$"),
    plot_name="Linear regression prediction",
)

# Compare different values of the model parameter
line_list = [data_line, mle_prediction_line]
param_list = np.arange(1, 11, 1) / 10
cp = plotting.ColourPicker(param_list.size, cyclic=False)
for i, param in enumerate(param_list):
    model._params = np.array(param).reshape(model._params.shape)
    Ytest = model.predict(Xtest)
    line_list.append(
        plotting.Line(Xtest, Ytest, c=cp(i), label="Theta = %.1f" % param)
    )
plotting.plot(
    *line_list,
    axis_properties=plotting.AxisProperties("$x$", "$y$"),
    plot_name="Comparing different values for $\\theta$",
    legend_properties=plotting.LegendProperties(),
)

# Modify the training targets and re-run
y_new = np.array(y)
y_new[-2] += 20
model = LinearRegression()
model.estimate_ml(X, y_new)
mle_prediction = model.predict(Xtest)
plotting.plot(
    plotting.Line(X, y_new, marker="+", ms=10, ls="", c="b"),
    plotting.Line(Xtest, mle_prediction, c="r"),
    axis_properties=plotting.AxisProperties("$x$", "$y_{new}$"),
    plot_name="Linear regression prediction with modified data",
)

# Add offsets and make predictions
y_new = y + 2
model = LinearRegression()
model.estimate_ml(X, y_new)
mle_prediction = model.predict(Xtest)
plotting.plot(
    plotting.Line(X, y_new, marker="+", ms=10, ls="", c="b"),
    plotting.Line(Xtest, mle_prediction, c="r"),
    axis_properties=plotting.AxisProperties("$x$", "$y_{new}$"),
    plot_name="Linear regression prediction with offset",
)

# Make predictions with affine features
model = LinearRegression(features=Affine())
model.estimate_ml(X, y_new)
mle_prediction = model.predict(Xtest)
plotting.plot(
    plotting.Line(X, y_new, marker="+", ms=10, ls="", c="b"),
    plotting.Line(Xtest, mle_prediction, c="r"),
    axis_properties=plotting.AxisProperties("$x$", "$y_{new}$"),
    plot_name="Linear regression prediction with affine features",
)

# Compare different values of the model parameters with affine features
data_line = plotting.Line(
    X,
    y_new,
    marker="+",
    ms=10,
    ls="",
    c="b",
    label="Data",
    zorder=30,
)
mle_prediction_line = plotting.Line(
    Xtest,
    mle_prediction,
    c="r",
    label="MLE estimate",
    zorder=20,
)
line_list = [data_line, mle_prediction_line]
param_list = [[a, b] for a in [0.3, 0.5, 0.6] for b in [1, 2, 3]]
cp = plotting.ColourPicker(len(param_list), cyclic=False)
for i, param in enumerate(param_list):
    model._params = np.array(param).reshape(model._params.shape)
    Ytest = model.predict(Xtest)
    line_list.append(
        plotting.Line(Xtest, Ytest, c=cp(i), label="Theta = %s" % param)
    )
plotting.plot(
    *line_list,
    axis_properties=plotting.AxisProperties("$x$", "$y$"),
    plot_name="Comparing different values for $\\theta$ with affine features",
    legend_properties=plotting.LegendProperties(),
)

# Nonlinear features
y = np.array([10.05, 1.5, -1.234, 0.02, 8.03]).reshape(-1,1)
data_line = plotting.Line(
    X,
    y,
    marker="+",
    ls="",
    c="b",
    label="Data",
    zorder=30,
)
plotting.plot(
    data_line,
    axis_properties=plotting.AxisProperties("$x$", "$y$"),
    plot_name="Linear regression dataset for nonlinear features",
)

for degree in range(2, 8):
    model = LinearRegression(features=Polynomial(degree=degree))
    model.estimate_ml(X, y)
    mle_prediction = model.predict(Xtest)
    mle_prediction_line = plotting.Line(
        Xtest,
        mle_prediction,
        c="r",
        label="MLE estimate",
        zorder=20,
    )
    plotting.plot(
        data_line,
        mle_prediction_line,
        axis_properties=plotting.AxisProperties("$x$", "$y$"),
        plot_name=(
            "Linear regression prediction with polynomial degree = %i"
            % degree
        ),
    )
