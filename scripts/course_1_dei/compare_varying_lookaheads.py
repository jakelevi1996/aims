import os
import numpy as np
import __init__
import data
import scripts.course_1_dei.gp_utils
import plotting

sotonmet = data.Sotonmet()

make_gif = False
filename_list = []

gp_name = "sum_opt"
g = scripts.course_1_dei.gp_utils.gp_dict[gp_name]

if make_gif:
    condition_boundary_array = np.linspace(0.1, max(sotonmet.t_train), 101)
    output_dir = os.path.join(
        scripts.course_1_dei.gp_utils.RESULTS_DIR,
        "varying_lookahead_gif",
    )
else:
    condition_boundary_array = np.arange(1, max(sotonmet.t_train))
    output_dir = os.path.join(
        scripts.course_1_dei.gp_utils.RESULTS_DIR,
        "varying_lookahead",
    )

t_ind = 0
for condition_boundary in condition_boundary_array:

    while (
        t_ind < sotonmet.t_train.size
        and sotonmet.t_train[t_ind] < condition_boundary
    ):
        g.condition(sotonmet.t_train[t_ind], sotonmet.y_train[t_ind])
        t_ind += 1

    y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)

    output_filename = plotting.plot(
        *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
        *scripts.course_1_dei.gp_utils.get_gp_prediction_lines(
            sotonmet.t_pred,
            y_pred_mean,
            y_pred_std,
        ),
        plotting.HVSpan(
            xlims=[0, condition_boundary],
            color="b",
            alpha=0.3,
            label="Conditioning set",
        ),
        plot_name=(
            "Lookahead, GP = %r, boundary = %.2f days"#
            % (gp_name, condition_boundary)
        ),
        dir_name=output_dir,
        axis_properties=scripts.course_1_dei.gp_utils.AXIS_PROPERTIES,
        legend_properties=plotting.LegendProperties(),
    )
    filename_list.append(output_filename)

if make_gif:
    plotting.make_gif(
        *filename_list,
        output_name="sequential_predictions_%s" % gp_name,
        output_dir=output_dir,
        frame_duration_ms=(10000 / condition_boundary_array.size),
    )
