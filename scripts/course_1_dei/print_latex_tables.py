import numpy as np
import __init__
import data
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()
gp_name_list = ["sqe_1", "sqe_2", "sqe_opt"]
gp_list = [
    scripts.course_1_dei.gp_utils.gp_dict[gp_name]
    for gp_name in gp_name_list
]
for g in gp_list:
    g.condition(sotonmet.t_train, sotonmet.y_train)

print("\nTable: SQE GPs\n")
table = [
    ["Hyperparameter"] + [
        gp_name.replace("_", "\\_") for gp_name in gp_name_list
    ],
    ["$c$"] + ["%.4f" % g._prior_mean_func._offset for g in gp_list],
    ["$\\lambda$"] + ["%.4f" % g._kernel_func._length_scale for g in gp_list],
    ["$k$"] + ["%.4f" % g._kernel_func._kernel_scale for g in gp_list],
    ["$\\sigma$"] + ["%.4f" % np.sqrt(g._noise_var) for g in gp_list],
]
print(table)
for row in table:
    print(" & ".join(row), end=" \\\\\n")

print("\nTable: SQE metrics\n")
table = [
    ["Metric"] + [
        gp_name.replace("_", "\\_") for gp_name in gp_name_list
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
print(table)
for row in table:
    print(" & ".join(row), end=" \\\\\n")
