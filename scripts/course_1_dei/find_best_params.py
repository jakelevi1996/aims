import time
import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

g = gp.GaussianProcess(
    prior_mean_func=gp.mean.Constant(3),
    kernel_func=gp.kernel.SquaredExponential(0.3, 10),
    noise_std=1,
)

g.optimise_hyperparameters(sotonmet.t_train, sotonmet.y_train)

g.decondition()
g.condition(sotonmet.t_train, sotonmet.y_train)
y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)
print(g.log_marginal_likelihood())

scripts.course_1_dei.gp_utils.plot_gp(
    g,
    sotonmet,
    plot_name="Data and optimised GP predictions, GP = %r" % g,
)
