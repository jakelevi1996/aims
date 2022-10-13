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

g = gp.GaussianProcess(
    prior_mean_func=mean.Constant(3),
    kernel_func=kernel.SquaredExponential(length_scale=0.1, kernel_scale=1),
    noise_std=0.001,
)
num_prior_samples = 5
prior_samples = [g.sample_prior(t_pred) for _ in range(num_prior_samples)]

cp = plotting.ColourPicker(num_prior_samples)
prior_sample_lines = [
    plotting.Line(t_pred, y_sample, c=cp(i), alpha=0.5, zorder=30)
    for i, y_sample in enumerate(prior_samples)
]
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
    *prior_sample_lines,
    plot_name="Samples from GP prior vs data",
)
