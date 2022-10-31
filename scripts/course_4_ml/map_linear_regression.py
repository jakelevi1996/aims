import numpy as np
import __init__
import linear_regression
import plotting
import util

# define the function we wish to estimate later
def g(x, sigma):
    p = np.hstack([x**0, x**1, np.sin(x)])
    w = np.array([-1.0, 0.1, 1.0]).reshape(-1,1)
    return p @ w + sigma*np.random.normal(size=x.shape)

# Generate some data
sigma = 1.0 # noise standard deviation
alpha = 1.0 # standard deviation of the parameter prior
N = 20

np.random.seed(42)

X = (np.random.rand(N)*10.0 - 5.0).reshape(-1,1)
y = g(X, sigma) # training targets

Xtest = np.linspace(-5,5,100).reshape(-1,1)
ytest = g(Xtest, sigma)

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

y_pred_mle = mle_model.predict(Xtest)
y_pred_map = map_model.predict(Xtest)

plotting.plot(
    plotting.Line(X, y, marker="+", c="b", ls="", label="Training data"),
    plotting.Line(Xtest, g(Xtest, 0), c="g", label="Ground truth function"),
    plotting.Line(Xtest, y_pred_mle, c="r", label="MLE prediction"),
    plotting.Line(Xtest, y_pred_map, c="m", label="MAP prediction"),
    axis_properties=plotting.AxisProperties("$x$", "$y$", ylim=[-5, 5]),
    plot_name="MAP vs MLE predictions, polynomial degree = 8",
    legend_properties=plotting.LegendProperties(),
)

util.numpy_set_print_options()
print("MLE:", mle_model, "MAP:", map_model, sep="\n")

# Plot RMSE vs polynomial order
K_max = 12 # this is the maximum degree of polynomial we will consider
assert(K_max < N)

rmse_mle = []
rmse_map = []

def rmse(y, ypred):
    return np.sqrt(np.mean(np.square(y - ypred)))

for k in range(K_max+1):
    mle_model = linear_regression.LinearRegression(
        features=linear_regression.features.Polynomial(degree=k),
    )
    map_model = linear_regression.LinearRegression(
        features=linear_regression.features.Polynomial(degree=k),
    )
    mle_model.estimate_ml(X, y)
    map_model.estimate_map(X, y, sigma, alpha)

    y_pred_mle = mle_model.predict(Xtest)
    y_pred_map = map_model.predict(Xtest)

    rmse_mle.append(rmse(ytest, y_pred_mle))
    rmse_map.append(rmse(ytest, y_pred_map))

plotting.plot(
    plotting.Line(rmse_mle, c="r", label="Maximum likelihood"),
    plotting.Line(rmse_map, c="m", label="MAP"),
    axis_properties=plotting.AxisProperties(
        xlabel="degree of polynomial",
        ylabel="RMSE",
        log_yscale=True,
    ),
    plot_name="RMSE vs degree polynomial, comparing MLE vs MAP",
    legend_properties=plotting.LegendProperties(),
)

# Plot predictions of best polynomial order
K = 4 # polynomial degree

# feature matrix
mle_model = linear_regression.LinearRegression(
    features=linear_regression.features.Polynomial(degree=K),
)
mle_model.estimate_ml(X, y)
map_model = linear_regression.LinearRegression(
    features=linear_regression.features.Polynomial(degree=K),
)
map_model.estimate_map(X, y, sigma, alpha)

y_pred_mle = mle_model.predict(Xtest)
y_pred_map = map_model.predict(Xtest)

plotting.plot(
    plotting.Line(X, y, marker="+", c="b", ls="", label="Training data"),
    plotting.Line(Xtest, g(Xtest, 0), c="g", label="Ground truth function"),
    plotting.Line(Xtest, y_pred_mle, c="r", label="MLE prediction"),
    plotting.Line(Xtest, y_pred_map, c="m", label="MAP prediction"),
    axis_properties=plotting.AxisProperties("$x$", "$y$"),
    plot_name="MAP vs MLE predictions, polynomial degree = 4",
    legend_properties=plotting.LegendProperties(),
)

# Find best value of alpha
alpha_list = np.exp(np.linspace(*np.log([0.1, 10])))
rmse_list = []
for alpha in alpha_list:
    map_model = linear_regression.LinearRegression(
        features=linear_regression.features.Polynomial(degree=4),
    )
    map_model.estimate_map(X, y, sigma, alpha)
    y_pred_map = map_model.predict(Xtest)
    rmse_list.append(rmse(ytest, y_pred_map))

plotting.plot(
    plotting.Line(alpha_list, rmse_list, c="b"),
    axis_properties=plotting.AxisProperties(
        xlabel="$\\alpha$",
        ylabel="RMSE",
        log_xscale=True,
        log_yscale=True,
    ),
    plot_name="RMSE vs prior standard deviation",
)
best_rmse = min(rmse_list)
best_alpha = alpha_list[rmse_list.index(best_rmse)]
print("Best alpha = %s, best RMSE = %s" % (best_alpha, best_rmse))

# Plot predictions of best polynomial order and best alpha
K = 4 # polynomial degree

mle_model = linear_regression.LinearRegression(
    features=linear_regression.features.Polynomial(degree=K),
)
mle_model.estimate_ml(X, y)
map_model = linear_regression.LinearRegression(
    features=linear_regression.features.Polynomial(degree=K),
)
map_model.estimate_map(X, y, sigma, best_alpha)

y_pred_mle = mle_model.predict(Xtest)
y_pred_map = map_model.predict(Xtest)

plotting.plot(
    plotting.Line(X, y, marker="+", c="b", ls="", label="Training data"),
    plotting.Line(Xtest, g(Xtest, 0), c="g", label="Ground truth function"),
    plotting.Line(Xtest, y_pred_mle, c="r", label="MLE prediction"),
    plotting.Line(Xtest, y_pred_map, c="m", label="MAP prediction"),
    axis_properties=plotting.AxisProperties("$x$", "$y$"),
    plot_name=(
        "MAP vs MLE predictions, polynomial degree = 4, "
        "$\\alpha_{MAP} = %.3f$"
        % best_alpha
    ),
    legend_properties=plotting.LegendProperties(),
)
