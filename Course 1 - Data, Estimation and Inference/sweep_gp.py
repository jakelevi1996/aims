import os
import numpy as np
import __init__
import data
import gp
import mean
import kernel
import sweep
import plotting

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sotonmet = data.Sotonmet()

class _GpSweep(sweep.Experiment):
    def get_gp(self, **kwargs):
        raise NotImplementedError()

    def run(self, **kwargs):
        gp = self.get_gp(**kwargs)
        gp.condition(sotonmet.t_train, sotonmet.y_train)
        return gp.log_marginal_likelihood()

class Periodic(_GpSweep):
    def get_gp(self, offset, period, length_scale, kernel_scale, noise_std):
        return gp.GaussianProcess(
            prior_mean_func=mean.Constant(offset),
            kernel_func=kernel.Periodic(period, length_scale, kernel_scale),
            noise_std=noise_std,
        )

sweeper = sweep.ParamSweeper(
    experiment=Periodic(),
    n_repeats=1,
    higher_is_better=True,
    verbose=False,
)
def add_log_range_param(sweeper, name, default, scale_factor=10, num=25):
    val_range = sweep.get_range(
        val_lo=default / scale_factor,
        val_hi=default * scale_factor,
        val_num=num,
        log_space=True,
    )
    param = sweep.Parameter(name, default, val_range, log_x_axis=True)
    sweeper.add_parameter(param)

sweeper.add_parameter(sweep.Parameter("offset", 3, np.arange(0, 4, 0.5)))
best_period = 0.514954586260453
add_log_range_param(sweeper, "period", best_period, scale_factor=4, num=101)
add_log_range_param(sweeper, "length_scale", 2.9)
add_log_range_param(sweeper, "kernel_scale", 6.1)
add_log_range_param(sweeper, "noise_std", 0.17)
optimal_param_dict = sweeper.find_best_parameters()

output_dir = os.path.join(CURRENT_DIR, "Results", "Param sweep", "Periodic")
sweeper.plot("Periodic kernel", output_dir)

g = Periodic().get_gp(**optimal_param_dict)
g.condition(sotonmet.t_train, sotonmet.y_train)
y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)

print(g)
print(g.log_marginal_likelihood())
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
    plot_name="Predictions after parameter sweep",
    axis_properties=plotting.AxisProperties(ylim=[0, 6])
)
sweeper.tighten_ranges()
optimal_param_dict = sweeper.find_best_parameters()
sweeper.plot(
    "Periodic kernel (tightened range)",
    os.path.join(output_dir, "tightened_range"),
)
g = Periodic().get_gp(**optimal_param_dict)
g.condition(sotonmet.t_train, sotonmet.y_train)
print(g)
print(g.log_marginal_likelihood())
