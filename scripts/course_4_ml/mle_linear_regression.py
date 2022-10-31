import numpy as np
import __init__
import linear_regression
import plotting

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
model = linear_regression.LinearRegression()
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
model = linear_regression.LinearRegression()
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
model = linear_regression.LinearRegression()
model.estimate_ml(X, y_new)
mle_prediction = model.predict(Xtest)
plotting.plot(
    plotting.Line(X, y_new, marker="+", ms=10, ls="", c="b"),
    plotting.Line(Xtest, mle_prediction, c="r"),
    axis_properties=plotting.AxisProperties("$x$", "$y_{new}$"),
    plot_name="Linear regression prediction with offset",
)

# Make predictions with affine features
model = linear_regression.LinearRegression(
    features=linear_regression.features.Affine(),
)
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
    model = linear_regression.LinearRegression(
        features=linear_regression.features.Polynomial(degree=degree),
    )
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

# Evaluating the Quality of the Model
rng = np.random.default_rng(0)

def f(x):
    return np.cos(x) + 0.2*rng.normal(size=(x.shape))

X = np.linspace(-4,4,20).reshape(-1,1)
y = f(X)
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
    plot_name="Noisy cosine dataset",
)

Xtest = np.linspace(-5,5,100).reshape(-1,1)
ytest = f(Xtest) # ground-truth y-values

for degree in range(2, 13):
    model = linear_regression.LinearRegression(
        features=linear_regression.features.Polynomial(degree=degree),
    )
    model.estimate_ml(X, y)
    mle_prediction = model.predict(Xtest)
    test_line = plotting.Line(
        Xtest,
        ytest,
        c="g",
        label="Test data",
        zorder=10,
    )
    mle_prediction_line = plotting.Line(
        Xtest,
        mle_prediction,
        c="r",
        label="MLE estimate",
        zorder=20,
    )
    plotting.plot(
        data_line,
        test_line,
        mle_prediction_line,
        axis_properties=plotting.AxisProperties("$x$", "$y$"),
        plot_name=(
            "Noisy cosine dataset with polynomial degree = %i"
            % degree
        ),
        legend_properties=plotting.LegendProperties(),
    )

# Plot RMSE vs polynomial degree
def rmse(y, ypred):
    return np.sqrt(np.mean(np.square(y - ypred)))

K_max = 20
rmse_train = []
rmse_test = []

for k in range(K_max+1):
    model = linear_regression.LinearRegression(
        features=linear_regression.features.Polynomial(degree=k),
    )
    model.estimate_ml(X, y)
    mle_prediction_train = model.predict(X)
    mle_prediction_test = model.predict(Xtest)

    rmse_train.append(rmse(y, mle_prediction_train))
    rmse_test.append(rmse(ytest, mle_prediction_test))

plotting.plot(
    plotting.Line(rmse_train, c="b", label="RMSE on training data"),
    plotting.Line(rmse_test, c="r", label="RMSE on test data"),
    axis_properties=plotting.AxisProperties(
        xlabel="degree of polynomial",
        ylabel="RMSE",
        log_yscale=True,
    ),
    legend_properties=plotting.LegendProperties(),
    plot_name="RMSE vs degree of polynomial for training and test data"
)
