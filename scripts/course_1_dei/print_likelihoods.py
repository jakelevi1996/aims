import numpy as np
import __init__
import data
import gp
import plotting
import scripts.course_1_dei.gp_utils

sotonmet = data.Sotonmet()

for gp_name in ["sqe_1", "sqe_2"]:
    g = scripts.course_1_dei.gp_utils.gp_dict[gp_name]
    g.condition(sotonmet.t_train, sotonmet.y_train)

    print(gp_name)
    print(g)
    print("RMSE (train) = %f" % g.rmse(sotonmet.t_train, sotonmet.y_train))
    print("RMSE (truth) = %f" % g.rmse(sotonmet.t_truth, sotonmet.y_truth))
    print("Log marginal likelihood = %f" % g.log_marginal_likelihood())
    print(
        "Log predictive likelihood = %f"
        % g.log_predictive_likelihood(sotonmet.t_truth, sotonmet.y_truth)
    )
    print(
        "Log predictive likelihood (train) = %f"
        % g.log_predictive_likelihood(sotonmet.t_train, sotonmet.y_train)
    )
    log_lik_list = []
    for _ in range(5):
        batch_inds = np.random.choice(
            sotonmet.n_truth,
            sotonmet.n_train,
            replace=False,
        )
        log_lik = g.log_predictive_likelihood(
            sotonmet.t_truth[batch_inds],
            sotonmet.y_truth[batch_inds],
        )
        log_lik_list.append(log_lik)

    print(
        "Log predictive likelihoods (truth subsets) = %s\n"
        % log_lik_list
    )

print("Markdown table:")
print("GP name | GP description | Log marginal likelihood")
print("--- | --- | ---")
for gp_name in ["sqe_1", "sqe_2", "sqe_opt"]:
    g = scripts.course_1_dei.gp_utils.gp_dict[gp_name]
    g.condition(sotonmet.t_train, sotonmet.y_train)
    print("`%s` | `%s` | %f" % (gp_name, g, g.log_marginal_likelihood()))
