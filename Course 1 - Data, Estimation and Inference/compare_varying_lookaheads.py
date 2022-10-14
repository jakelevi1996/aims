import os
import numpy as np
import __init__
import data
import gp
import mean
import kernel
import plotting

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sotonmet = data.Sotonmet()
t_pred = np.linspace(-1, 6, 1000)

for condition_boundary in np.arange(1, max(sotonmet.t_train)):
    t_train = sotonmet.t_train[sotonmet.t_train < condition_boundary]
    y_train = sotonmet.y_train[sotonmet.t_train < condition_boundary]

    g = gp.GaussianProcess(
        prior_mean_func=mean.Constant(3),
        kernel_func=kernel.SquaredExponential(0.08666701, 0.65337298),
        noise_std=0.02931095,
    )
    g.condition(t_train, y_train)
    y_pred_mean, y_pred_std = g.predict(t_pred)

    num_posterior_samples = 5
    posterior_samples = g.sample_posterior(t_pred, num_posterior_samples)

    posterior_sample_lines = [
        plotting.Line(
            t_pred,
            posterior_samples[:, i],
            c="r",
            alpha=0.2,
            zorder=30,
        )
        for i in range(num_posterior_samples)
    ]

    plotting.plot(
        *sotonmet.get_train_test_plot_lines(),
        plotting.Line(t_pred, y_pred_mean, c="r", zorder=40),
        plotting.FillBetween(
            t_pred,
            y_pred_mean + 2 * y_pred_std,
            y_pred_mean - 2 * y_pred_std,
            color="r",
            lw=0,
            alpha=0.2,
            zorder=30,
        ),
        *posterior_sample_lines,
        plotting.HVLine(v=condition_boundary, ls="--", c="r"),
        plot_name="Lookahead, boundary = %.1f days" % condition_boundary,
        dir_name=os.path.join(CURRENT_DIR, "Results", "Varying lookahead"),
        axis_properties=plotting.AxisProperties(ylim=[0, 6]),
    )
