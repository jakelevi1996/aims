import numpy as np
import scipy.optimize
import __init__
import data
import gp
import mean
import kernel
import plotting

data_dict, num_columns, num_rows = data.load_dict()
assert num_columns == 19
assert num_rows == 1258
t_pred, t_data, y_data = data.parse_dict(data_dict)
assert t_pred.size == 1258
assert t_data.size == 917
assert y_data.size == 917

def negative_log_marginal_likelihood(log_args):
    args = np.exp(log_args)
    g = gp.GaussianProcess(
        prior_mean_func=mean.Constant(args[0]),
        kernel_func=kernel.SquaredExponential(args[1], args[2]),
        noise_std=args[3],
    )
    g.condition(t_data, y_data)

    nlml = -g.log_marginal_likelihood()
    print(args, nlml)
    return nlml

initial_params = np.log([3, .3, 10, 1])
result = scipy.optimize.minimize(
    negative_log_marginal_likelihood,
    initial_params,
    options={"maxiter": 30},
)
prior_mean, length_scale, kernel_scale, noise_std = np.exp(result.x)
print(prior_mean, length_scale, kernel_scale, noise_std)

g = gp.GaussianProcess(
    prior_mean_func=mean.Constant(prior_mean),
    kernel_func=kernel.SquaredExponential(length_scale, length_scale),
    noise_std=noise_std,
)
g.condition(t_data, y_data)
y_pred_mean, y_pred_std = g.predict(t_pred)

plotting.plot(
    plotting.Line(
        t_data,
        y_data,
        c="k",
        ls="",
        marker="o",
        alpha=0.5,
        zorder=20,
    ),
    plotting.Line(
        t_pred,
        data.parse_column(data_dict, "True tide height (m)"),
        c="k",
        ls="",
        marker="x",
        alpha=0.5,
        zorder=20,
    ),
    plotting.Line(t_pred, y_pred_mean, c="r", zorder=40),
    plotting.FillBetween(
        t_pred,
        y_pred_mean + 2*y_pred_std,
        y_pred_mean - 2*y_pred_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
    ),
    plot_name="Data and optimised GP predictions",
)
