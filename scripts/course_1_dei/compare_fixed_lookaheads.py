import os
import numpy as np
import __init__
import data
import scripts.course_1_dei.gp_utils
import plotting

DAYS_PER_MINUTE = 1 / (60 * 24)
T_STEP = 10 * DAYS_PER_MINUTE
T_MAX = 7

sotonmet = data.Sotonmet()

for lookahead_minutes in [0, 5, 50, 500, 5000]:
    lookahead_days = lookahead_minutes * DAYS_PER_MINUTE
    t_pred_array = np.linspace(lookahead_days, T_MAX, 1000)
    t_train_ind = 0
    y_pred_mean_list = []
    y_pred_std_list = []

    g = scripts.course_1_dei.gp_utils.get_optimal_gp()
    g.condition(
        sotonmet.t_train[t_train_ind],
        sotonmet.y_train[t_train_ind],
    )
    t_train_ind += 1

    for t_pred in t_pred_array:
        while (
            t_train_ind < sotonmet.t_train.size
            and sotonmet.t_train[t_train_ind] + lookahead_days <= t_pred
        ):
            g.condition(
                sotonmet.t_train[t_train_ind],
                sotonmet.y_train[t_train_ind],
            )
            t_train_ind += 1

        y_pred_mean, y_pred_std = g.predict(t_pred)
        y_pred_mean_list.append(y_pred_mean)
        y_pred_std_list.append(y_pred_std)

    plot_name = (
        "Sequential prediction with fixed minimum lookahead = %i minutes"
        % lookahead_minutes
    )
    plotting.plot(
        *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
        *scripts.course_1_dei.gp_utils.get_gp_prediction_lines(
            t_pred_array,
            np.reshape(y_pred_mean_list, -1),
            np.reshape(y_pred_std_list, -1),
        ),
        plotting.HVLine(
            v=lookahead_days,
            c="r",
            ls="--",
            label="Start of lookahead predictions",
        ),
        plot_name=plot_name,
        dir_name=os.path.join(
            scripts.course_1_dei.gp_utils.RESULTS_DIR,
            "fixed_lookahead"
        ),
        axis_properties=scripts.course_1_dei.gp_utils.AXIS_PROPERTIES,
        legend_properties=plotting.LegendProperties(),
    )
