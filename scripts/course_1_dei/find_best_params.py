import time
import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

for gp_name in ["sqe_2", "per_1"]:
    g = scripts.course_1_dei.gp_utils.gp_dict[gp_name]
    g.optimise_hyperparameters(sotonmet.t_train, sotonmet.y_train)

    g.decondition()
    g.condition(sotonmet.t_train, sotonmet.y_train)
    y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)
    print(
        "Final GP = %r\nFinal log likelihood = %f"
        % (g, g.log_marginal_likelihood())
    )
