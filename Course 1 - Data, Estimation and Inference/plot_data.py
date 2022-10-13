import numpy as np
import __init__
import util
import plotting
import data
import gp
import mean
import kernel

data_dict, num_columns, num_rows = data.load_dict()
assert num_columns == 19
assert num_rows == 1258
t_pred, t_data, y_data = data.parse_dict(data_dict)
assert t_pred.size == 1258
assert t_data.size == 917
assert y_data.size == 917

data_line = plotting.Line(
    t_data,
    y_data,
    c="k",
    ls="",
    marker="o",
    alpha=0.5,
    zorder=20,
)
plotting.plot(data_line, plot_name="Tide height (m) vs time (days)")

g = gp.GaussianProcess(mean.ZeroMean(), kernel.SquaredExponential(0.3, 10), 1)
num_prior_samples = 5
prior_sample_lines = [
    plotting.Line(t_pred, g.sample_prior(t_pred), c="b", alpha=0.5)
    for _ in range(num_prior_samples)
]
plotting.plot(
    data_line,
    *prior_sample_lines,
    plot_name="Samples from GP prior vs data",
)

g.condition(t_data, y_data)
y_pred_mean, y_pred_std = g.predict(t_pred)

plotting.plot(
    data_line,
    plotting.Line(t_pred, y_pred_mean, c="r", zorder=40),
    plotting.FillBetween(
        t_pred,
        y_pred_mean + y_pred_std,
        y_pred_mean - y_pred_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
    ),
    plot_name="Data and GP predictions",
)
