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
    plot_name = "GP predictions and predictive samples, GP = %r" % gp_name

    scripts.course_1_dei.gp_utils.plot_gp(
        g,
        sotonmet,
        plot_name,
        axis_properties=plotting.AxisProperties(
            xlabel=scripts.course_1_dei.gp_utils.X_LABEL,
            ylabel=scripts.course_1_dei.gp_utils.Y_LABEL,
            xlim=scripts.course_1_dei.gp_utils.X_LIM,
        ),
    )
