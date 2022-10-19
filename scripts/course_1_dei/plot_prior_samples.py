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
    num_prior_samples = 5
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
        plot_name="Samples from GP prior, GP = %r" % g,
        dir_name=scripts.course_1_dei.gp_utils.RESULTS_DIR,
        axis_properties=scripts.course_1_dei.gp_utils.AXIS_PROPERTIES,
        legend_properties=plotting.LegendProperties(),
    )
