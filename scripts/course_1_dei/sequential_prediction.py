import numpy as np
import __init__
import data
import gp
import mean
import kernel
import plotting

data_dict, num_columns, num_rows = data.load_dict()
assert num_columns == 19
assert num_rows == 1258
t_pred, t_data, y_data = data.parse_dict(data_dict)
assert t_pred.size == 1258
assert t_data.size == 917
assert y_data.size == 917

g = gp.GaussianProcess(
    prior_mean_func=mean.Constant(3),
    kernel_func=kernel.SquaredExponential(0.3, 10),
    noise_std=1,
)

for i, [t, y], in enumerate(zip(t_data, y_data)):
    print("\rConditioning on point %i..." % (i + 1), end="", flush=True)
    g.condition(t, y)

print("\nFinished sequential conditioning")
y_pred_mean_sequential, y_pred_std_sequential = g.predict(t_pred)

plotting.plot(
    *data.get_train_test_plot_lines(t_data, y_data, t_pred, data_dict),
    plotting.Line(t_pred, y_pred_mean_sequential, c="r", zorder=40),
    plotting.FillBetween(
        t_pred,
        y_pred_mean_sequential + y_pred_std_sequential,
        y_pred_mean_sequential - y_pred_std_sequential,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
    ),
    plot_name="Data and sequential GP predictions",
)

g.decondition()
g.condition(t_data, y_data)
y_pred_mean, y_pred_std = g.predict(t_pred)
print(
    "Max mean error: %f"
    % np.max(np.abs(y_pred_mean_sequential - y_pred_mean))
)
print(
    "Max std error: %f"
    % np.max(np.abs(y_pred_std_sequential - y_pred_std))
)
