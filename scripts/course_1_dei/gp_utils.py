import os
import __init__
import gp
import plotting
import data
import scripts.course_1_dei

RESULTS_DIR = os.path.join(scripts.course_1_dei.CURRENT_DIR, "Results")

X_LABEL = "Time (days)"
Y_LABEL = "Tide height (m)"
X_LIM = data.sotonmet.T_LIM
Y_LIM = [0, 6]
AXIS_PROPERTIES = plotting.AxisProperties(
    X_LABEL,
    Y_LABEL,
    xlim=X_LIM,
    ylim=Y_LIM,
)

gp_dict = {
    "sqe_1": gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(3),
        kernel_func=gp.kernel.SquaredExponential(0.1, 1),
        noise_std=0.001,
    ),
    "sqe_2": gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(3),
        kernel_func=gp.kernel.SquaredExponential(0.3, 10),
        noise_std=1,
    ),
    "sqe_opt": gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(offset=2.9904846516133974),
        kernel_func=gp.kernel.SquaredExponential(
            length_scale=0.08665037458315064,
            kernel_scale=0.6522383851241347,
        ),
        noise_std=0.02930675775064153,
    ),
    "per_1": gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(offset=2.9904846516133974),
        kernel_func=gp.kernel.Periodic(
            period=0.5,
            length_scale=0.08665037458315064,
            kernel_scale=0.6522383851241347,
        ),
        noise_std=0.02930675775064153,
    ),
    "per_opt": gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(offset=2.994526707406642),
        kernel_func=gp.kernel.Periodic(
            period=0.5149342760919302,
            length_scale=1.2264134716027426,
            kernel_scale=1.0346460845353729,
        ),
        noise_std=0.17334345487465788,
    ),
    "sum_1": gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(offset=2.994526707406642),
        kernel_func=gp.kernel.Sum(
            gp.kernel.SquaredExponential(
                length_scale=0.08665037458315064,
                kernel_scale=0.6522383851241347,
            ),
            gp.kernel.Periodic(
                period=0.5149342760919302,
                length_scale=1.2264134716027426,
                kernel_scale=1.0346460845353729,
            ),
        ),
        noise_std=0.17334345487465788,
    ),
    "prod_1": gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(offset=2.994526707406642),
        kernel_func=gp.kernel.Product(
            gp.kernel.SquaredExponential(
                length_scale=0.08665037458315064,
                kernel_scale=0.6522383851241347,
            ),
            gp.kernel.Periodic(
                period=0.5149342760919302,
                length_scale=1.2264134716027426,
                kernel_scale=1.0346460845353729,
            ),
        ),
        noise_std=0.17334345487465788,
    ),
    "sum_opt": gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(offset=3),
        kernel_func=gp.kernel.Sum(
            gp.kernel.SquaredExponential(
                length_scale=0.06917512071945595,
                kernel_scale=0.029895345214372513,
            ),
            gp.kernel.Periodic(
                period=0.514954586260453,
                length_scale=0.6453392671906066,
                kernel_scale=0.5962321347520633,
            ),
        ),
        noise_std=0.02871806422941413,
    ),
    "prod_opt": gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(offset=2.9285714285714284),
        kernel_func=gp.kernel.Product(
            gp.kernel.SquaredExponential(
                length_scale=0.7880754080416588,
                kernel_scale=5.538930746200925,
            ),
            gp.kernel.Periodic(
                period=0.5082224844864489,
                length_scale=0.7270981370167336,
                kernel_scale=0.0974970247641131,
            ),
        ),
        noise_std=0.02905329635947378,
    ),
}

def get_optimal_gp():
    g = gp_dict["sum_opt"]
    g.decondition()
    return g

def plot_gp(
    g,
    dataset,
    plot_name,
    dir_name=RESULTS_DIR,
    axis_properties=AXIS_PROPERTIES,
):
    g.decondition()
    g.condition(dataset.t_train, dataset.y_train)
    y_pred_mean, y_pred_std = g.predict(dataset.t_pred)

    num_posterior_samples = 5
    posterior_samples = g.sample_posterior(
        dataset.t_pred,
        num_posterior_samples,
    )

    plotting.plot(
        *get_dataset_lines(dataset),
        *get_gp_prediction_lines(dataset.t_pred, y_pred_mean, y_pred_std),
        *get_gp_posterior_sample_lines(dataset.t_pred, posterior_samples),
        plot_name=plot_name,
        dir_name=dir_name,
        axis_properties=axis_properties,
        legend_properties=plotting.LegendProperties(),
    )

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
        label="GP predictive mean",
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

def get_gp_posterior_sample_lines(x, posterior_samples):
    posterior_sample_lines = [
        plotting.Line(
            x,
            posterior_samples[:, i],
            c="k",
            alpha=0.1,
            zorder=30,
        )
        for i in range(posterior_samples.shape[1])
    ]
    label_line = plotting.Line(
        [],
        [],
        c="k",
        label="GP predictive sample",
    )
    return posterior_sample_lines + [label_line]
