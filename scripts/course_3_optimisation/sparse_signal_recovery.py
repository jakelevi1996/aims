import numpy as np
import cvxpy as cp
import __init__
import plotting
import util

def project_l1_norm(a, norm=1):
    x = cp.Variable(a.size)
    s = cp.Variable(a.size)
    objective = cp.Minimize(cp.sum_squares(a - x))
    constraints = [-s <= x, x <= s, s >= 0, cp.sum(s) <= norm]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return x.value

class _Loss:
    def __call__(self, x):
        raise NotImplementedError

    def dydx(self, x):
        raise NotImplementedError

class Huber(_Loss):
    def __call__(self, x):
        x_abs = np.abs(x)
        return np.sum(np.where(x_abs < 1, x*x, 2 * x_abs - 1))

    def dydx(self, x):
        return np.clip(2 * x, -2, 2)

    def __repr__(self):
        return "Huber loss"

class L1(_Loss):
    def __call__(self, x):
        return np.sum(np.abs(x))

    def dydx(self, x):
        return np.sign(x)

    def __repr__(self):
        return "L1 loss"

class SumSquares(_Loss):
    def __call__(self, x):
        return np.sum(np.square(x))

    def dydx(self, x):
        return 2 * x

    def __repr__(self):
        return "Sum of squares loss"

# Generate random signal
rng = np.random.default_rng(0)
n = 1000
num_spikes = 10
t = np.arange(n)
x_original = np.zeros(n)
spike_inds = rng.integers(0, n, num_spikes)
x_original[spike_inds] = rng.standard_normal(num_spikes)

# Samples
n_samples = 100
t_samples = np.arange(n_samples)
A = rng.standard_normal([n_samples, n])
y = A @ x_original

for loss in [Huber(), L1(), SumSquares()]:
    step_size = 1e-3
    x = np.zeros(x_original.shape)
    error_list = []
    timer = util.Timer()

    for i in range(150):
        print("\ri = %i... " % i, end="", flush=True)
        dphi_dx = A.T @ loss.dydx(A @ x - y)
        x += (-step_size) * dphi_dx
        x = project_l1_norm(x, 10)
        error_list.append(np.linalg.norm(x - x_original))

    timer.print_time_taken()
    plotting.plot(
        plotting.Line(np.arange(len(error_list)), error_list, c="b"),
        plot_name="Error vs iteration (%s)" % loss,
    )
    plotting.plot(
        plotting.Line(t, x_original, c="r", label="Ground truth"),
        plotting.Line(t, x, c="b", label="Prediction"),
        plot_name="Ground truth vs prediction (%s)" % loss,
        legend_properties=plotting.LegendProperties(),
    )
    plotting.plot(
        plotting.Line(t_samples, y, c="r", label="Observed samples"),
        plotting.Line(t_samples, A @ x, c="b", label="Predicted samples"),
        plot_name="Observed vs predicted samples (%s)" % loss,
        legend_properties=plotting.LegendProperties(),
    )
