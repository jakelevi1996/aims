import __init__
import data
import plotting

data_dict, num_columns, num_rows = data.load_dict()
assert num_columns == 19
assert num_rows == 1258
t_pred, t_data, y_data = data.parse_dict(data_dict)
assert t_pred.size == 1258
assert t_data.size == 917
assert y_data.size == 917

plotting.plot(
    plotting.Line(
        t_data,
        y_data,
        c="k",
        ls="",
        marker="o",
        alpha=0.5,
        zorder=20,
    ),
    plot_name="Tide height (m) vs time (days)",
)

y_pred_mean = data.parse_column(
    data_dict,
    "Independent tide height prediction (m)",
)
y_pred_std = data.parse_column(
    data_dict,
    "Independent tide height deviation (m)",
)

plotting.plot(
    plotting.Line(
        t_data,
        y_data,
        c="k",
        ls="",
        marker="o",
        alpha=0.5,
        zorder=20,
    ),
    plotting.Line(
        t_pred,
        data.parse_column(data_dict, "True tide height (m)"),
        c="k",
        ls="",
        marker="x",
        alpha=0.5,
        zorder=20,
    ),
    plotting.Line(t_pred, y_pred_mean, c="r", zorder=40),
    plotting.FillBetween(
        t_pred,
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
