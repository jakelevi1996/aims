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
    g.condition(sotonmet.t_train, sotonmet.y_train)
    y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)

    plotting.plot(
        *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
        *scripts.course_1_dei.gp_utils.get_gp_prediction_lines(
            sotonmet.t_pred,
            y_pred_mean,
            y_pred_std,
        ),
        plot_name="Data and GP predictions, GP = %r" % g,
        dir_name=scripts.course_1_dei.gp_utils.RESULTS_DIR,
        axis_properties=scripts.course_1_dei.gp_utils.AXIS_PROPERTIES,
        legend_properties=plotting.LegendProperties(),
    )
