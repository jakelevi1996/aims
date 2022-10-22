import numpy as np
import __init__
import data
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

print("Conditioning all models...")
for g in scripts.course_1_dei.gp_utils.gp_dict.values():
    g.condition(sotonmet.t_train, sotonmet.y_train)

# Hyperparameters
print("\nTable: SQE GPs\n")
sqe_name_list = ["sqe_1", "sqe_2", "sqe_opt"]
sqe_list = [
    scripts.course_1_dei.gp_utils.gp_dict[name]
    for name in sqe_name_list
]
table = [
    ["Hyperparameter"] + [
        name.replace("_", "\\_") for name in sqe_name_list
    ],
    ["$c$"]        + ["%.4f" % g._prior_mean_func._offset   for g in sqe_list],
    ["$\\lambda$"] + ["%.4f" % g._kernel_func._length_scale for g in sqe_list],
    ["$k$"]        + ["%.4f" % g._kernel_func._kernel_scale for g in sqe_list],
    ["$\\sigma$"]  + ["%.4f" % np.sqrt(g._noise_var)        for g in sqe_list],
]

for row in table:
    print(" & ".join(row), end=" \\\\\n")

print("\nTable: PER GPs\n")
per_name_list = ["per_1", "per_opt"]
per_list = [
    scripts.course_1_dei.gp_utils.gp_dict[name]
    for name in per_name_list
]
table = [
    ["Hyperparameter"] + [
        name.replace("_", "\\_") for name in per_name_list
    ],
    ["$c$"]        + ["%.4f" % g._prior_mean_func._offset   for g in per_list],
    ["$\\lambda$"] + ["%.4f" % g._kernel_func._length_scale for g in per_list],
    ["$k$"]        + ["%.4f" % g._kernel_func._kernel_scale for g in per_list],
    ["$\\sigma$"]  + ["%.4f" % np.sqrt(g._noise_var)        for g in per_list],
    ["$T$"]        + ["%.4f" % (np.pi / g._kernel_func._angular_freq) for g in per_list],
]

for row in table:
    print(" & ".join(row), end=" \\\\\n")

# Metrics
print("\nTable: metrics\n")
gp_name_list = scripts.course_1_dei.gp_utils.gp_dict.keys()
gp_list = scripts.course_1_dei.gp_utils.gp_dict.values()
table = [
    ["Metric"] + [
        gp_name.replace("_", "\\_")
        for gp_name in scripts.course_1_dei.gp_utils.gp_dict.keys()
    ],
    ["RMSE (train)"] + [
        "%.4f" % g.rmse(sotonmet.t_train, sotonmet.y_train)
        for g in gp_list
    ],
    ["RMSE (truth)"] + [
        "%.4f" % g.rmse(sotonmet.t_truth, sotonmet.y_truth)
        for g in gp_list
    ],
    ["LML"] + [
        "%.1f" % g.log_marginal_likelihood()
        for g in gp_list
    ],
    ["LPL (train)"] + [
        "%.1f" % g.log_predictive_likelihood(
            sotonmet.t_train,
            sotonmet.y_train,
        )
        for g in gp_list
    ],
    ["LPL (truth)"] + [
        "%.1f" % g.log_predictive_likelihood(
            sotonmet.t_truth,
            sotonmet.y_truth,
        )
        for g in gp_list
    ],
]

for row in table:
    print(" & ".join(row), end=" \\\\\n")
