import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

g = gp.GaussianProcess(
    prior_mean_func=gp.mean.Constant(3),
    kernel_func=gp.kernel.SquaredExponential(0.08666701, 0.65337298),
    noise_std=0.02931095,
)
g.condition(sotonmet.t_train, sotonmet.y_train)
y_pred_mean, y_pred_std = g.predict(sotonmet.t_pred)

num_posterior_samples = 5
posterior_samples = g.sample_posterior(sotonmet.t_pred, num_posterior_samples)

plotting.plot(
    *scripts.course_1_dei.gp_utils.get_dataset_lines(sotonmet),
    *scripts.course_1_dei.gp_utils.get_gp_prediction_lines(
        sotonmet.t_pred,
        y_pred_mean,
        y_pred_std,
    ),
    *scripts.course_1_dei.gp_utils.get_gp_posterior_sample_lines(
        sotonmet.t_pred,
        posterior_samples,
    ),
    plot_name="Data and GP predictions and posterior samples",
    dir_name=scripts.course_1_dei.gp_utils.RESULTS_DIR,
    axis_properties=scripts.course_1_dei.gp_utils.AXIS_PROPERTIES,
    legend_properties=plotting.LegendProperties(),
)
