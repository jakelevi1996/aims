import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

num_prior_samples = 5

for gp_name in ["sqe_1", "sqe_2"]:
    g = scripts.course_1_dei.gp_utils.gp_dict[gp_name]
    prior_samples = g.sample_prior(sotonmet.t_pred, num_prior_samples)

    cp = plotting.ColourPicker(num_prior_samples)
    prior_sample_lines = [
        plotting.Line(
            sotonmet.t_pred,
            prior_samples[:, i],
            c=cp(i),
            alpha=0.5,
            zorder=30,
            label="Prior sample %i" % (i + 1)
        )
        for i in range(num_prior_samples)
    ]
    plotting.plot(
        *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
        *prior_sample_lines,
        plot_name="Samples from GP prior, GP = %r" % gp_name,
        dir_name=scripts.course_1_dei.gp_utils.RESULTS_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel=scripts.course_1_dei.gp_utils.X_LABEL,
            ylabel=scripts.course_1_dei.gp_utils.Y_LABEL,
            xlim=scripts.course_1_dei.gp_utils.X_LIM,
        ),
        legend_properties=plotting.LegendProperties(),
    )
