import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

gp_list = [
    gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(3),
        kernel_func=gp.kernel.SquaredExponential(0.1, 1),
        noise_std=0.001,
    ),
    gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(3),
        kernel_func=gp.kernel.SquaredExponential(0.3, 10),
        noise_std=1,
    ),
]

for g in gp_list:
    plot_name = "Data and GP predictions and posterior samples, GP = %r" % g
    scripts.course_1_dei.gp_utils.plot_gp(g, sotonmet, plot_name)
