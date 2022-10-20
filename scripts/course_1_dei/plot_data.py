import __init__
import data
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

axis_properties = plotting.AxisProperties(
    xlabel=scripts.course_1_dei.gp_utils.X_LABEL,
    ylabel=scripts.course_1_dei.gp_utils.Y_LABEL,
    ylim=scripts.course_1_dei.gp_utils.Y_LIM,
)

plotting.plot(
    *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
    plot_name="Sotonmet data",
    dir_name=scripts.course_1_dei.gp_utils.RESULTS_DIR,
    axis_properties=axis_properties,
    legend_properties=plotting.LegendProperties(),
)

y_pred_mean = sotonmet.get_column_data(
    "Independent tide height prediction (m)",
)
y_pred_std = sotonmet.get_column_data(
    "Independent tide height deviation (m)",
)

plotting.plot(
    *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
    *scripts.course_1_dei.gp_utils.get_gp_prediction_lines(
        sotonmet.t_truth,
        y_pred_mean,
        y_pred_std,
    ),
    plot_name="Data and independent GP predictions",
    dir_name=scripts.course_1_dei.gp_utils.RESULTS_DIR,
    axis_properties=axis_properties,
    legend_properties=plotting.LegendProperties(),
)
