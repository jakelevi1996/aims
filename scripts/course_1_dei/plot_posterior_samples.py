import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

g = gp.GaussianProcess(
    prior_mean_func=gp.mean.Constant(3),
    kernel_func=gp.kernel.SquaredExponential(0.08666701, 0.65337298),
    noise_std=0.02931095,
)

plot_name = "Data and GP predictions and posterior samples"
scripts.course_1_dei.gp_utils.plot_gp(g, sotonmet, plot_name)
