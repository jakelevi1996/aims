import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

gp_name_list = ["sqe_1", "sqe_2"]

for gp_name in gp_name_list:
    g = scripts.course_1_dei.gp_utils.gp_dict[gp_name]
    g.condition(sotonmet.t_train, sotonmet.y_train)
    y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)

    plotting.plot(
        *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
        *scripts.course_1_dei.gp_utils.get_gp_prediction_lines(
            sotonmet.t_pred,
            y_pred_mean,
            y_pred_std,
        ),
        plot_name="GP predictions, GP = %r" % gp_name,
        dir_name=scripts.course_1_dei.gp_utils.RESULTS_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel=scripts.course_1_dei.gp_utils.X_LABEL,
            ylabel=scripts.course_1_dei.gp_utils.Y_LABEL,
            xlim=scripts.course_1_dei.gp_utils.X_LIM,
        ),
        legend_properties=plotting.LegendProperties(),
    )
