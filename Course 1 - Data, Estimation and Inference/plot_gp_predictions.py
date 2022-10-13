import numpy as np
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

g = gp.GaussianProcess(mean.ZeroMean(), kernel.SquaredExponential(0.3, 10), 1)
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
