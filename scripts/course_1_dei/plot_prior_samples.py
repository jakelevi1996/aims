import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

g = gp.GaussianProcess(
    prior_mean_func=gp.mean.Constant(3),
    kernel_func=gp.kernel.SquaredExponential(length_scale=0.1, kernel_scale=1),
    noise_std=0.001,
)
num_prior_samples = 5
prior_samples = g.sample_prior(sotonmet.t_pred, num_prior_samples)

cp = plotting.ColourPicker(num_prior_samples)
prior_sample_lines = [
    plotting.Line(
        sotonmet.t_pred,
        prior_samples[:, i],
        c=cp(i),
        alpha=0.5,
        zorder=30,
    )
    for i in range(num_prior_samples)
]
plotting.plot(
    *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
    *prior_sample_lines,
    plot_name="Samples from GP prior vs data",
)
