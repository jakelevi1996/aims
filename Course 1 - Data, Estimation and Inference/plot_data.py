import time
import calendar
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

t_str_list = data_dict["Reading Date and Time (ISO)"]
y_str_list = data_dict["Tide height (m)"]

t_format = "%Y-%m-%dT%H:%M:%S"
days_per_second = 1 / (60 * 60 * 24)

get_timestamp = lambda t_str: calendar.timegm(time.strptime(t_str, t_format))
t0 = get_timestamp(t_str_list[0])
t_list = [
    (get_timestamp(t_str) - t0) * days_per_second
    for t_str in t_str_list
]

has_data_list = [len(y) > 0 for y in y_str_list]

y_data = [
    float(y_str)
    for y_str, has_data in zip(y_str_list, has_data_list)
    if has_data
]
t_data = [t for t, has_data in zip(t_list, has_data_list) if has_data]
t_pred = [t for t, has_data in zip(t_list, has_data_list) if not has_data]

data_line = plotting.Line(t_data, y_data, c="b", ls="-", marker="o", alpha=0.5)
plotting.plot(data_line, plot_name="Tide height (m) vs time (days)")
pred_lines = [plotting.HVLine(v=t, c="r", alpha=0.2) for t in t_pred]
plotting.plot(
    data_line,
    *pred_lines,
    plot_name="Tide height (m) vs time (days), including missing data points",
)

g = gp.GaussianProcess(mean.ZeroMean(), kernel.SquaredExponential(1, 1), 0.1)
num_prior_samples = 5
prior_sample_lines = [
    plotting.Line(t_list, g.sample_prior(np.array(t_list)), c="b", alpha=0.5)
    for _ in range(num_prior_samples)
]
plotting.plot(
    data_line,
    *prior_sample_lines,
    plot_name="Samples from GP prior vs data",
)

g.condition(np.array(t_data), np.array(y_data))
y_pred_mean, y_pred_std = g.predict(np.array(t_list))

plotting.plot(
    plotting.Line(t_data, y_data, c="k", ls="", marker="o", alpha=0.5),
    plotting.Line(t_list, y_pred_mean, c="r"),
    plotting.FillBetween(
        t_list,
        y_pred_mean + y_pred_std,
        y_pred_mean - y_pred_std,
        color="r",
        alpha=0.2,
    ),
    plot_name="Data and GP predictions",
)
