import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

g = gp.GaussianProcess(
    prior_mean_func=gp.mean.Constant(3),
    kernel_func=gp.kernel.SquaredExponential(
        0.0866675466933244,
        0.6540971841037699,
    ),
    noise_std=0.029309042867821246,
)

plot_name = "Predictions from optimal GP with squared exponential kernel"
scripts.course_1_dei.gp_utils.plot_gp(g, sotonmet, plot_name)

g = scripts.course_1_dei.gp_utils.get_optimal_gp()
plot_name = "Predictions from optimal GP with sum of kernels"
scripts.course_1_dei.gp_utils.plot_gp(g, sotonmet, plot_name)
