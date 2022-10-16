import numpy as np
import __init__
import data
import gp
import mean
import kernel
import plotting

sotonmet = data.Sotonmet()
t_pred = np.linspace(-1, 6, 1000)

g_list = [
    gp.GaussianProcess(
        prior_mean_func=mean.Constant(3),
        kernel_func=kernel.SquaredExponential(0.3, 10),
        noise_std=1,
    ),
    gp.GaussianProcess(
        prior_mean_func=mean.Constant(3),
        kernel_func=kernel.SquaredExponential(0.08666701, 0.65337298),
        noise_std=0.02931095,
    ),
]
for g in g_list:
    g.condition(sotonmet.t_train, sotonmet.y_train)
    y_grad_mean, y_grad_std = g.predict_gradient(t_pred)
    t_grad = (t_pred[:-1] + t_pred[1:]) / 2

    plotting.plot(
        plotting.Line(t_grad, y_grad_mean, c="r", zorder=40),
        plotting.FillBetween(
            t_grad,
            y_grad_mean + y_grad_std,
            y_grad_mean - y_grad_std,
            color="r",
            lw=0,
            alpha=0.2,
            zorder=30,
        ),
        plot_name="GP gradients, GP = %r" % g,
    )
