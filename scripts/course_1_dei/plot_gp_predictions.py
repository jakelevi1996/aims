import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

g = gp.GaussianProcess(
    prior_mean_func=gp.mean.Constant(3),
    kernel_func=gp.kernel.SquaredExponential(0.3, 10),
    noise_std=1,
)
g.condition(sotonmet.t_train, sotonmet.y_train)
y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)

plotting.plot(
    plotting.Line(
        sotonmet.t_train,
        sotonmet.y_train,
        c="k",
        ls="",
        marker="o",
        alpha=0.5,
        zorder=20,
        label="Training data",
    ),
    plotting.Line(
        sotonmet.t_truth,
        sotonmet.y_truth,
        c="k",
        ls="",
        marker="x",
        alpha=0.5,
        zorder=20,
        label="Ground truth data",
    ),
    plotting.Line(
        sotonmet.t_pred,
        y_pred_mean,
        c="r",
        zorder=40,
        label="GP mean prediction",
    ),
    plotting.FillBetween(
        sotonmet.t_pred,
        y_pred_mean + y_pred_std,
        y_pred_mean - y_pred_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
    ),
    plotting.FillBetween(
        [],
        [],
        [],
        color="r",
        lw=0,
        alpha=0.5,
        label="$\\pm \\sigma$",
    ),
    plotting.FillBetween(
        sotonmet.t_pred,
        y_pred_mean + 2 * y_pred_std,
        y_pred_mean - 2 * y_pred_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
        label="$\\pm 2 \\sigma$",
    ),
    plot_name="Data and GP predictions",
    dir_name=scripts.course_1_dei.gp_utils.RESULTS_DIR,
    axis_properties=plotting.AxisProperties("Time (days)", "Tide height (m)"),
    legend_properties=plotting.LegendProperties(),
)

print("Log marginal likelihood = %f" % g.log_marginal_likelihood())
print("RMSE (train) = %f" % g.rmse(sotonmet.t_train, sotonmet.y_train))
print("RMSE (truth) = %f" % g.rmse(sotonmet.t_truth, sotonmet.y_truth))
print(
    "Log predictive likelihood = %f"
    % g.log_predictive_likelihood(sotonmet.t_truth, sotonmet.y_truth)
)
print(
    "Log predictive likelihood (train)= %f"
    % g.log_predictive_likelihood(sotonmet.t_train, sotonmet.y_train)
)
for _ in range(5):
    batch_inds = np.random.choice(
        sotonmet.n_truth,
        sotonmet.n_train,
        replace=False,
    )
    print(
        "Log predictive likelihood (truth subset)= %f"
        % g.log_predictive_likelihood(
            sotonmet.t_truth[batch_inds],
            sotonmet.y_truth[batch_inds],
        )
    )
