import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

for gp_name in ["sqe_opt", "sum_opt"]:
    g = scripts.course_1_dei.gp_utils.gp_dict[gp_name]
    plot_name = "Optimised GP predictions, GP = %r" % gp_name
    scripts.course_1_dei.gp_utils.plot_gp(g, sotonmet, plot_name)
