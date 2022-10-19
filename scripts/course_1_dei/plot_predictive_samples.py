import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

gp_list = [
    scripts.course_1_dei.gp_utils.gp_dict["sqe_1"],
    scripts.course_1_dei.gp_utils.gp_dict["sqe_2"],
]

for g in gp_list:
    plot_name = "Data and GP predictions and predictive samples, GP = %r" % g
    scripts.course_1_dei.gp_utils.plot_gp(g, sotonmet, plot_name)
