import os
import numpy as np
import __init__
import data
import gp
import sweep
import plotting
import scripts.course_1_dei.gp_utils

OUTPUT_DIR = os.path.join(
    scripts.course_1_dei.gp_utils.RESULTS_DIR,
    "param_sweep",
)

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

    def _add_param(self, name, default, val_range):
        self._sweeper.add_parameter(
            sweep.Parameter(
                name,
                default,
                val_range,
                plot_axis_properties=plotting.AxisProperties(
                    xlabel=name,
                    ylabel="Log marginal likelihood",
                ),
            )
        )

    def _add_log_range_param(self, name, default, scale_factor=10, num=25):
        val_range = sweep.get_range(
            val_lo=default / scale_factor,
            val_hi=default * scale_factor,
            val_num=num,
            log_space=True,
        )
        param = sweep.Parameter(
            name,
            default,
            val_range,
            plot_axis_properties=plotting.AxisProperties(
                xlabel=name,
                ylabel="Log marginal likelihood",
                log_xscale=True,
            ),
        )
        self._sweeper.add_parameter(param)

class SquaredExponential(_GpSweep):
    def _add_parameters(self):
        self._add_param("offset", 3, np.arange(0, 4, 0.5))
        self._add_log_range_param("length_scale", 0.0866675466933244)
        self._add_log_range_param("kernel_scale", 0.6540971841037699)
        self._add_log_range_param("noise_std", 0.029309042867821246)

    def _get_gp(self, offset, length_scale, kernel_scale, noise_std):
        return gp.GaussianProcess(
            prior_mean_func=gp.mean.Constant(offset),
            kernel_func=gp.kernel.SquaredExponential(
                length_scale,
                kernel_scale,
            ),
            noise_std=noise_std,
        )

class Periodic(_GpSweep):
    def _add_parameters(self):
        self._add_param("offset", 3, np.arange(0, 4, 0.5))
        self._add_log_range_param(
            "period",
            0.514954586260453,
            scale_factor=4,
            num=101,
        )
        self._add_log_range_param("length_scale", 1.2507974381079883)
        self._add_log_range_param("kernel_scale", 0.480600322135365)
        self._add_log_range_param("noise_std", 0.17313920673121166)

    def _get_gp(self, offset, period, length_scale, kernel_scale, noise_std):
        return gp.GaussianProcess(
            prior_mean_func=gp.mean.Constant(offset),
            kernel_func=gp.kernel.Periodic(
                period,
                length_scale,
                kernel_scale,
            ),
            noise_std=noise_std,
        )

class Sum(_GpSweep):
    def _add_parameters(self):
        self._add_param("offset", 3, np.arange(0, 4, 0.5))
        self._add_log_range_param("sqe_length_scale", 0.06917512071945595)
        self._add_log_range_param("sqe_kernel_scale", 0.029895345214372513)
        self._add_log_range_param("period", 0.514954586260453, scale_factor=4)
        self._add_log_range_param("per_length_scale", 0.6512752017924203)
        self._add_log_range_param("per_kernel_scale", 0.13082827454160073)
        self._add_log_range_param("noise_std", 0.02871806422941413)

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
            prior_mean_func=gp.mean.Constant(offset),
            kernel_func=gp.kernel.Sum(
                gp.kernel.SquaredExponential(
                    sqe_length_scale,
                    sqe_kernel_scale,
                ),
                gp.kernel.Periodic(
                    period,
                    per_length_scale,
                    per_kernel_scale,
                ),
            ),
            noise_std=noise_std,
        )

class Product(_GpSweep):
    def _add_parameters(self):
        self._add_param("offset", 3, np.arange(0, 4, 0.5))
        self._add_log_range_param("sqe_length_scale", 0.7880754080416588)
        self._add_log_range_param("sqe_kernel_scale", 1.4457480054362188)
        self._add_log_range_param("period", 0.5082224844864489, scale_factor=4)
        self._add_log_range_param("per_length_scale", 0.7405355857351571)
        self._add_log_range_param("per_kernel_scale", 0.09929886031253296)
        self._add_log_range_param("noise_std", 0.02905329635947378)

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
            prior_mean_func=gp.mean.Constant(offset),
            kernel_func=gp.kernel.Product(
                gp.kernel.SquaredExponential(
                    sqe_length_scale,
                    sqe_kernel_scale,
                ),
                gp.kernel.Periodic(
                    period,
                    per_length_scale,
                    per_kernel_scale,
                ),
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

    print(g)
    print(g.log_marginal_likelihood())

    scripts.course_1_dei.gp_utils.plot_gp(
        g,
        sotonmet,
        plot_name=(
            "Predictions for %s kernel after parameter sweep"
            % sweeper.name
        ),
        dir_name=sweeper.output_dir,
    )
