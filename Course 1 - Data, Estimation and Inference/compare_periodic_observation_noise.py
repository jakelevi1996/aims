import numpy as np
import __init__
import data
import gp
import mean
import kernel
import plotting

sotonmet = data.Sotonmet()
observation_noise_std = 0.17334367049162486

g = gp.GaussianProcess(
    prior_mean_func=mean.Constant(offset=2.9942653663746848),
    kernel_func=kernel.Periodic(
        period=0.5149342508262474,
        length_scale=1.2259554067739233,
        kernel_scale=0.45733875534737783,
    ),
    noise_std=observation_noise_std,
)
g.condition(sotonmet.t_train, sotonmet.y_train)
y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)

plotting.plot(
    *sotonmet.get_train_test_plot_lines(),
    plotting.Line(sotonmet.t_pred, y_pred_mean, c="r", zorder=40),
    plotting.FillBetween(
        sotonmet.t_pred,
        y_pred_mean + 2 * y_pred_std,
        y_pred_mean - 2 * y_pred_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
    ),
    plot_name="Data and periodic GP predictions without observation noise",
)
y_pred_std += observation_noise_std
plotting.plot(
    *sotonmet.get_train_test_plot_lines(),
    plotting.Line(sotonmet.t_pred, y_pred_mean, c="r", zorder=40),
    plotting.FillBetween(
        sotonmet.t_pred,
        y_pred_mean + 2 * y_pred_std,
        y_pred_mean - 2 * y_pred_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
    ),
    plot_name="Data and periodic GP predictions with observation noise",
)
