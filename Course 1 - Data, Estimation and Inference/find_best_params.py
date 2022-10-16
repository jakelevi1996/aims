import time
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
        kernel_func=kernel.Periodic(0.5, 0.3, 10),
        noise_std=1,
    ),
    gp.GaussianProcess(
        prior_mean_func=mean.Constant(3),
        kernel_func=kernel.Sum(
            kernel.SquaredExponential(0.3, 10),
            kernel.Periodic(0.5, 0.3, 10),
        ),
        noise_std=1,
    ),
]
running_time_list = []

for g in g_list:
    t0 = time.perf_counter()
    g.optimise_hyperparameters(sotonmet.t_train, sotonmet.y_train)
    running_time_list.append(time.perf_counter() - t0)

for g in g_list:
    print("\n%r" % g)

    g.decondition()
    g.condition(sotonmet.t_train, sotonmet.y_train)
    y_pred_mean, y_pred_std = g.predict(t_pred)
    print(g.log_marginal_likelihood())

    plotting.plot(
        *sotonmet.get_train_test_plot_lines(),
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
        plot_name="Data and optimised GP predictions, GP = %r" % g,
        axis_properties=plotting.AxisProperties(ylim=[0, 6]),
    )

print("\nTime taken to optimise each GP: %s" % running_time_list)
print("Total time taken: %s" % sum(running_time_list))
