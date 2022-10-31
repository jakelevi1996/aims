import numpy as np
import __init__
import plotting

class LinearRegression:
    def __init__(self):
        self._params = None

    def estimate_ml(self, X, y):
        """
        X: N x D matrix of training inputs
        y: N x 1 vector of training targets/observations
        Calculates maximum likelihood parameters (D x 1) and stores in _params
        """
        self._params, *_ = np.linalg.lstsq(X, y, rcond=None)

    def predict(self, X):
        """
        Xtest: K x D matrix of test inputs
        returns: prediction of f(Xtest); K x 1 vector
        """
        if self._params is None:
            raise ValueError("Must estimate parameters before predicting")

        return X @ self._params

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
ml_prediction = model.predict(Xtest)

# plot
mle_prediction_line = plotting.Line(
    Xtest,
    ml_prediction,
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
y[-2] += 20
model = LinearRegression()
model.estimate_ml(X, y)
ml_prediction = model.predict(Xtest)
plotting.plot(
    plotting.Line(X, y, marker="+", ms=10, ls="", c="b"),
    plotting.Line(Xtest, ml_prediction, c="r"),
    axis_properties=plotting.AxisProperties("$x$", "$y$"),
    plot_name="Linear regression prediction with modified data",
)
