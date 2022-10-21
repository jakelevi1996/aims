import os
import numpy as np
import __init__
import data
import scripts.course_1_dei.gp_utils
import plotting

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sotonmet = data.Sotonmet()

for condition_boundary in np.arange(1, max(sotonmet.t_train)):
    t_train = sotonmet.t_train[sotonmet.t_train < condition_boundary]
    y_train = sotonmet.y_train[sotonmet.t_train < condition_boundary]

    g = scripts.course_1_dei.gp_utils.get_optimal_gp()
    g.condition(t_train, y_train)
    y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)

    num_posterior_samples = 5
    posterior_samples = g.sample_posterior(
        sotonmet.t_pred,
        num_posterior_samples,
    )

    plotting.plot(
        *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
        *scripts.course_1_dei.gp_utils.get_gp_prediction_lines(
            sotonmet.t_pred,
            y_pred_mean,
            y_pred_std,
        ),
        *scripts.course_1_dei.gp_utils.get_gp_posterior_sample_lines(
            sotonmet.t_pred,
            posterior_samples,
        ),
        plotting.HVLine(
            v=condition_boundary,
            c="r",
            ls="--",
            label="Conditioning boundary",
        ),
        plot_name="Lookahead, boundary = %.1f days" % condition_boundary,
        dir_name=os.path.join(
            scripts.course_1_dei.gp_utils.RESULTS_DIR,
            "varying_lookahead"
        ),
        axis_properties=scripts.course_1_dei.gp_utils.AXIS_PROPERTIES,
        legend_properties=plotting.LegendProperties(),
    )
