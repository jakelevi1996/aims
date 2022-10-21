import os
import numpy as np
import __init__
import data
import scripts.course_1_dei.gp_utils
import plotting

sotonmet = data.Sotonmet()

for gp_name in ["sqe_opt", "prod_opt", "sum_opt"]:
    g = scripts.course_1_dei.gp_utils.gp_dict[gp_name]
    g.condition(sotonmet.t_train, sotonmet.y_train)
    y_grad_mean, y_grad_std = g.predict_gradient(sotonmet.t_pred)
    t_grad = (sotonmet.t_pred[:-1] + sotonmet.t_pred[1:]) / 2

    plotting.plot(
        *scripts.course_1_dei.gp_utils.get_gp_prediction_lines(
            t_grad,
            y_grad_mean,
            y_grad_std,
        ),
        plot_name="GP gradients, GP = %r" % gp_name,
        dir_name=os.path.join(
            scripts.course_1_dei.gp_utils.RESULTS_DIR,
            "gradients"
        ),
        axis_properties=plotting.AxisProperties(
                xlabel=scripts.course_1_dei.gp_utils.X_LABEL,
                ylabel="Gradient (metres/day)",
                xlim=scripts.course_1_dei.gp_utils.X_LIM,
        ),
        legend_properties=plotting.LegendProperties(),
    )
