import os
import __init__
import gp
import plotting
import scripts.course_1_dei

RESULTS_DIR = os.path.join(scripts.course_1_dei.CURRENT_DIR, "Results")

X_LABEL = "Time (days)"
Y_LABEL = "Tide height (m)"
AXIS_PROPERTIES = plotting.AxisProperties(X_LABEL, Y_LABEL)

def get_optimal_gp():
    g = gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(offset=3),
        kernel_func=gp.kernel.Sum(
            gp.kernel.SquaredExponential(
                length_scale=0.06917512071945595,
                kernel_scale=0.029895345214372513,
            ),
            gp.kernel.Periodic(
                period=0.514954586260453,
                length_scale=0.6512752017924203,
                kernel_scale=0.13082827454160073,
            ),
        ),
        noise_std=0.02871806422941413,
    )
    return g

def get_dataset_lines(dataset):
    train_line = plotting.Line(
        dataset.t_train,
        dataset.y_train,
        c="k",
        ls="",
        marker="o",
        alpha=0.5,
        zorder=20,
        label="Training data",
    )
    truth_line = plotting.Line(
        dataset.t_truth,
        dataset.y_truth,
        c="k",
        ls="",
        marker="x",
        alpha=0.5,
        zorder=20,
        label="Ground truth data",
    )
    return train_line, truth_line

def get_gp_prediction_lines(x, y_mean, y_std):
    mean_line = plotting.Line(
        x,
        y_mean,
        c="r",
        zorder=40,
        label="GP mean prediction",
    )
    pm_std_line = plotting.FillBetween(
        x,
        y_mean + y_std,
        y_mean - y_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
    )
    pm_2_std_line = plotting.FillBetween(
        x,
        y_mean + 2 * y_std,
        y_mean - 2 * y_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
        label="$\\pm 2 \\sigma$",
    )
    pm_std_label_line = plotting.FillBetween(
        [],
        [],
        [],
        color="r",
        lw=0,
        alpha=0.5,
        label="$\\pm \\sigma$",
    )
    return mean_line, pm_std_line, pm_std_label_line, pm_2_std_line
