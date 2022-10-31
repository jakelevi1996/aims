import numpy as np
import __init__
import linear_regression
import plotting
import util

rng = np.random.default_rng(1)

# define the function we wish to estimate later
def g(x, sigma):
    p = np.hstack([x**0, x**1, np.sin(x)])
    w = np.array([-1.0, 0.1, 1.0]).reshape(-1,1)
    return p @ w + sigma*rng.normal(size=x.shape)

# Generate some data
sigma = 1.0 # noise standard deviation
alpha = 1.0 # standard deviation of the parameter prior
N = 20

X = (rng.random(N)*10.0 - 5.0).reshape(-1,1)
y = g(X, sigma) # training targets

plotting.plot(
    plotting.Line(X, y, marker="+", c="b", ls=""),
    axis_properties=plotting.AxisProperties("$x$", "$y$"),
    plot_name="Data for MAP estimation"
)

# get the MAP estimate
K = 8 # polynomial degree

# feature matrix
mle_model = linear_regression.LinearRegression(
    features=linear_regression.features.Polynomial(degree=K),
)
mle_model.estimate_ml(X, y)
map_model = linear_regression.LinearRegression(
    features=linear_regression.features.Polynomial(degree=K),
)
map_model.estimate_map(X, y, sigma, alpha)

Xtest = np.linspace(-5,5,100).reshape(-1,1)
ytest = g(Xtest, sigma)

y_pred_mle = mle_model.predict(Xtest)
y_pred_map = map_model.predict(Xtest)

plotting.plot(
    plotting.Line(X, y, marker="+", c="b", ls="", label="Training data"),
    plotting.Line(Xtest, g(Xtest, 0), c="g", label="Ground truth function"),
    plotting.Line(Xtest, y_pred_mle, c="r", label="MLE prediction"),
    plotting.Line(Xtest, y_pred_map, c="m", label="MAP prediction"),
    axis_properties=plotting.AxisProperties("$x$", "$y$", ylim=[-5, 5]),
    plot_name="MAP vs MLE predictions",
    legend_properties=plotting.LegendProperties(),
)

util.numpy_set_print_options()
print("MLE:", mle_model, "MAP:", map_model, sep="\n")
