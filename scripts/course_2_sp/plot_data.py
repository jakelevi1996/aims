import os
import numpy as np
import __init__
import plotting

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QBO_PATH = os.path.join(CURRENT_DIR, "data", "qbo.txt")

with open(QBO_PATH) as f:
    data_line_list = f.read().split("\n")

qbo_table = [
    [float(s) for s in line.split()]
    for line in data_line_list
    if len(line) > 0
]
qbo_array = np.block(qbo_table)
t = np.arange(qbo_array.shape[0])

plotting.plot(
    plotting.Line(t, qbo_array[:, 0], c="r", label="Temperature stream 1"),
    plotting.Line(t, qbo_array[:, 1], c="g", label="Temperature stream 2"),
    plotting.Line(t, qbo_array[:, 2], c="b", label="Temperature stream 3"),
    plot_name="Quasi-Biennial Oscillation (QBO) temperature readings",
    axis_properties=plotting.AxisProperties(
        xlabel="Time (months)",
        ylabel="Temperature",
        ylim=[-600, 600]
    ),
    legend_properties=plotting.LegendProperties(),
)

def get_autocorrelation_coefficients_embedding(x, n_coeffs):
    embedding_matrix = x[
        np.flip(np.arange(n_coeffs))
        + np.arange(x.size - n_coeffs).reshape(-1, 1)
    ]
    coeffs, *_ = np.linalg.lstsq(embedding_matrix, x[n_coeffs:], rcond=None)
    return coeffs

coeffs = get_autocorrelation_coefficients_embedding(qbo_array[:, 0], 4)
print(coeffs)

def get_autocorrelation_coefficients_autocovariance(x, n_coeffs):
    # TODO: this function has a bug and doesn't give the correct answers
    x_zero_mean = x - np.mean(x)
    var = np.mean(x_zero_mean * x_zero_mean)
    auto_cov_list = [
        np.mean(x_zero_mean[i:] * x_zero_mean[:-i])
        for i in range(1, n_coeffs + 1)
    ]
    auto_cov = np.array(auto_cov_list)
    auto_cov_matrix = np.empty([n_coeffs, n_coeffs])
    diag_inds = np.arange(n_coeffs)
    auto_cov_matrix[diag_inds, diag_inds] = var
    for i in range(1, n_coeffs):
        auto_cov_matrix[i - 1, i:] = auto_cov[:-i]
        auto_cov_matrix[i:, i - 1] = auto_cov[:-i]

    coeffs = np.linalg.solve(auto_cov_matrix, auto_cov)
    return coeffs

coeffs = get_autocorrelation_coefficients_autocovariance(qbo_array[:, 0], 4)
print(coeffs)
