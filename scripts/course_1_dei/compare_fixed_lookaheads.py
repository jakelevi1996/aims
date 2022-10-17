import os
import numpy as np
import __init__
import data
import gp
import mean
import kernel
import plotting

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DAYS_PER_MINUTE = 1 / (60 * 24)
T_STEP = 10 * DAYS_PER_MINUTE
T_MAX = 7

sotonmet = data.Sotonmet()

for lookahead_minutes in [0, 5, 50, 500]:
    lookahead_days = lookahead_minutes * DAYS_PER_MINUTE
    t_pred_list = []
    y_pred_mean_list = []
    y_pred_std_list = []

    g = gp.GaussianProcess(
        prior_mean_func=mean.Constant(3),
        kernel_func=kernel.SquaredExponential(0.08666701, 0.65337298),
        noise_std=0.02931095,
    )

    t_previous = sotonmet.t_train[-1]
    for t, y in zip(sotonmet.t_train, sotonmet.y_train):
        t_pred = t + lookahead_days

        if t_pred > t_previous + T_STEP:
            for t_pred in np.arange(t_previous, t_pred, T_STEP):
                y_pred_mean, y_pred_std = g.predict(t_pred)

                t_pred_list.append(t_pred)
                y_pred_mean_list.append(y_pred_mean)
                y_pred_std_list.append(y_pred_std)

        g.condition(t, y)
        y_pred_mean, y_pred_std = g.predict(t_pred)

        t_pred_list.append(t_pred)
        y_pred_mean_list.append(y_pred_mean)
        y_pred_std_list.append(y_pred_std)
        t_previous = t_pred

    for t_pred in np.arange(t_previous, T_MAX, T_STEP):
        y_pred_mean, y_pred_std = g.predict(t_pred)

        t_pred_list.append(t_pred)
        y_pred_mean_list.append(y_pred_mean)
        y_pred_std_list.append(y_pred_std)

    plot_name = (
        "Sequential prediction with fixed minimum lookahead = %i minutes"
        % lookahead_minutes
    )
    y_pred_mean_array = np.reshape(y_pred_mean_list, -1)
    y_pred_std_array = np.reshape(y_pred_std_list, -1)
    plotting.plot(
        *sotonmet.get_train_test_plot_lines(),
        plotting.Line(t_pred_list, y_pred_mean_list, c="r", zorder=40),
        plotting.FillBetween(
            t_pred_list,
            y_pred_mean_array + 2 * y_pred_std_array,
            y_pred_mean_array - 2 * y_pred_std_array,
            color="r",
            lw=0,
            alpha=0.2,
            zorder=30,
        ),
        plot_name=plot_name,
        dir_name=os.path.join(CURRENT_DIR, "Results", "Fixed lookahead"),
        axis_properties=plotting.AxisProperties(ylim=[0, 6]),
    )
