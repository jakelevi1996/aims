import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

g = scripts.course_1_dei.gp_utils.gp_dict["sqe_opt"]

sotonmet_epistemic = data.Sotonmet()
day_2_3_inds = np.all(
    [
        sotonmet_epistemic.t_train >= 2,
        sotonmet_epistemic.t_train < 4,
    ],
    axis=0,
)
other_day_inds = np.logical_not(day_2_3_inds)
sotonmet_epistemic.t_train = sotonmet_epistemic.t_train[other_day_inds]
sotonmet_epistemic.y_train = sotonmet_epistemic.y_train[other_day_inds]

plot_name = "GP predictions with epistemic uncertainty"
scripts.course_1_dei.gp_utils.plot_gp(
    g,
    sotonmet_epistemic,
    plot_name,
    axis_properties=plotting.AxisProperties(
        xlabel=scripts.course_1_dei.gp_utils.X_LABEL,
        ylabel=scripts.course_1_dei.gp_utils.Y_LABEL,
        xlim=scripts.course_1_dei.gp_utils.X_LIM,
    ),
)

sotonmet_aleatoric = data.Sotonmet()
rng = np.random.default_rng()
sotonmet_aleatoric.y_train[day_2_3_inds] += rng.normal(
    size=sotonmet_aleatoric.y_train[day_2_3_inds].shape,
)

plot_name = "GP predictions with aleatoric uncertainty"
scripts.course_1_dei.gp_utils.plot_gp(
    g,
    sotonmet_aleatoric,
    plot_name,
    axis_properties=plotting.AxisProperties(
        xlabel=scripts.course_1_dei.gp_utils.X_LABEL,
        ylabel=scripts.course_1_dei.gp_utils.Y_LABEL,
        xlim=scripts.course_1_dei.gp_utils.X_LIM,
    ),
)
