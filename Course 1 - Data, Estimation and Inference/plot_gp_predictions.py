import numpy as np
import __init__
import data
import gp
import mean
import kernel
import plotting

sotonmet = data.Sotonmet()
t_pred = np.linspace(-1, 6, 1000)

g = gp.GaussianProcess(
    prior_mean_func=mean.Constant(3),
    kernel_func=kernel.SquaredExponential(0.3, 10),
    noise_std=1,
)
g.condition(sotonmet.t_train, sotonmet.y_train)
y_pred_mean, y_pred_std = g.predict(t_pred)

plotting.plot(
    *sotonmet.get_train_test_plot_lines(),
    plotting.Line(t_pred, y_pred_mean, c="r", zorder=40),
    plotting.FillBetween(
        t_pred,
        y_pred_mean + y_pred_std,
        y_pred_mean - y_pred_std,
        color="r",
        lw=0,
        alpha=0.2,
        zorder=30,
    ),
    plot_name="Data and GP predictions",
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
