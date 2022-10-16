import __init__
import data
import plotting

sotonmet = data.Sotonmet()

plotting.plot(
    *sotonmet.get_train_test_plot_lines(),
    plot_name="Tide height (m) vs time (days)",
)

y_pred_mean = sotonmet.get_column_data(
    "Independent tide height prediction (m)",
)
y_pred_std = sotonmet.get_column_data(
    "Independent tide height deviation (m)",
)

plotting.plot(
    *sotonmet.get_train_test_plot_lines(),
    plotting.Line(sotonmet.t_truth, y_pred_mean, c="r", zorder=40),
    plotting.FillBetween(
        sotonmet.t_truth,
        y_pred_mean + 2*y_pred_std,
        y_pred_mean - 2*y_pred_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
    ),
    plot_name="Data and independent GP predictions",
    axis_properties=plotting.AxisProperties(ylim=[0, 6]),
)
