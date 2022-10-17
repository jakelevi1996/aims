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
OUTPUT_DIR = os.path.join(CURRENT_DIR, "Results", "param_sweep")

sotonmet = data.Sotonmet()

class _GpSweep(sweep.Experiment):
    def _add_parameters(self):
        raise NotImplementedError()

    def _get_gp(self, **kwargs):
        raise NotImplementedError()

    def __init__(self):
        self.name = type(self).__name__
        self.output_dir = os.path.join(OUTPUT_DIR, self.name)

    def run(self, **kwargs):
        g = self._get_gp(**kwargs)
        g.condition(sotonmet.t_train, sotonmet.y_train)
        return g.log_marginal_likelihood()

    def find_best_parameters(self):
        self._sweeper = sweep.ParamSweeper(
            experiment=self,
            n_repeats=1,
            higher_is_better=True,
            verbose=False,
        )
        self._add_parameters()
        optimal_param_dict = self._sweeper.find_best_parameters()
        self._sweeper.plot(
            experiment_name="%s kernel" % self.name,
            output_dir=self.output_dir,
        )
        g = self._get_gp(**optimal_param_dict)
        return g

    def find_better_parameters(self):
        self._sweeper.tighten_ranges()
        optimal_param_dict = self._sweeper.find_best_parameters()
        self._sweeper.plot(
            experiment_name="%s kernel (tightened range)" % self.name,
            output_dir=os.path.join(self.output_dir, "tightened_range"),
        )
        g = self._get_gp(**optimal_param_dict)
        return g

    def _add_log_range_param(self, name, default, scale_factor=10, num=25):
        val_range = sweep.get_range(
            val_lo=default / scale_factor,
            val_hi=default * scale_factor,
            val_num=num,
            log_space=True,
        )
        param = sweep.Parameter(name, default, val_range, log_x_axis=True)
        self._sweeper.add_parameter(param)

class SquaredExponential(_GpSweep):
    def _add_parameters(self):
        self._sweeper.add_parameter(
            sweep.Parameter("offset", 3, np.arange(0, 4, 0.5))
        )
        self._add_log_range_param("length_scale", 0.087)
        self._add_log_range_param("kernel_scale", 0.65)
        self._add_log_range_param("noise_std", 0.029)

    def _get_gp(self, offset, length_scale, kernel_scale, noise_std):
        return gp.GaussianProcess(
            prior_mean_func=mean.Constant(offset),
            kernel_func=kernel.SquaredExponential(length_scale, kernel_scale),
            noise_std=noise_std,
        )

class Periodic(_GpSweep):
    def _add_parameters(self):
        self._sweeper.add_parameter(
            sweep.Parameter("offset", 3, np.arange(0, 4, 0.5))
        )
        self._add_log_range_param(
            "period",
            0.514954586260453,
            scale_factor=4,
            num=101,
        )
        self._add_log_range_param("length_scale", 2.9)
        self._add_log_range_param("kernel_scale", 6.1)
        self._add_log_range_param("noise_std", 0.17)

    def _get_gp(self, offset, period, length_scale, kernel_scale, noise_std):
        return gp.GaussianProcess(
            prior_mean_func=mean.Constant(offset),
            kernel_func=kernel.Periodic(period, length_scale, kernel_scale),
            noise_std=noise_std,
        )

class Sum(_GpSweep):
    def _add_parameters(self):
        self._sweeper.add_parameter(
            sweep.Parameter("offset", 3, np.arange(0, 4, 0.5))
        )
        self._add_log_range_param("sqe_length_scale", 0.087)
        self._add_log_range_param("sqe_kernel_scale", 0.65)
        self._add_log_range_param("period", 0.514954586260453, scale_factor=4)
        self._add_log_range_param("per_length_scale", 2.9)
        self._add_log_range_param("per_kernel_scale", 6.1)
        self._add_log_range_param("noise_std", 0.10)

    def _get_gp(
        self,
        offset,
        sqe_length_scale,
        sqe_kernel_scale,
        period,
        per_length_scale,
        per_kernel_scale,
        noise_std,
    ):
        return gp.GaussianProcess(
            prior_mean_func=mean.Constant(offset),
            kernel_func=kernel.Sum(
                kernel.SquaredExponential(sqe_length_scale, sqe_kernel_scale),
                kernel.Periodic(period, per_length_scale, per_kernel_scale),
            ),
            noise_std=noise_std,
        )

class Product(_GpSweep):
    def _add_parameters(self):
        self._sweeper.add_parameter(
            sweep.Parameter("offset", 3, np.arange(0, 4, 0.5))
        )
        self._add_log_range_param("sqe_length_scale", 0.087)
        self._add_log_range_param("sqe_kernel_scale", 0.065)
        self._add_log_range_param("period", 0.514954586260453, scale_factor=4)
        self._add_log_range_param("per_length_scale", 2.9)
        self._add_log_range_param("per_kernel_scale", 0.193)
        self._add_log_range_param("noise_std", 0.10)

    def _get_gp(
        self,
        offset,
        sqe_length_scale,
        sqe_kernel_scale,
        period,
        per_length_scale,
        per_kernel_scale,
        noise_std,
    ):
        return gp.GaussianProcess(
            prior_mean_func=mean.Constant(offset),
            kernel_func=kernel.Product(
                kernel.SquaredExponential(sqe_length_scale, sqe_kernel_scale),
                kernel.Periodic(period, per_length_scale, per_kernel_scale),
            ),
            noise_std=noise_std,
        )

sweeper_list = [
    SquaredExponential(),
    Periodic(),
    Sum(),
    Product(),
]

for sweeper in sweeper_list:
    g = sweeper.find_best_parameters()
    g.condition(sotonmet.t_train, sotonmet.y_train)

    print(g)
    print(g.log_marginal_likelihood())

    g = sweeper.find_better_parameters()
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
        plot_name=(
            "Predictions for %s kernel after parameter sweep"
            % sweeper.name
        ),
        dir_name=sweeper.output_dir,
        axis_properties=plotting.AxisProperties(ylim=[0, 6])
    )
