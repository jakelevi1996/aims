import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

g = scripts.course_1_dei.gp_utils.gp_dict["sqe_opt"]
plot_name = "Predictions from optimal GP with squared exponential kernel"
scripts.course_1_dei.gp_utils.plot_gp(g, sotonmet, plot_name)

g = scripts.course_1_dei.gp_utils.get_optimal_gp()
plot_name = "Predictions from optimal GP with sum of kernels"
scripts.course_1_dei.gp_utils.plot_gp(g, sotonmet, plot_name)
