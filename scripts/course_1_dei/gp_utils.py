import os
import __init__
import gp
import scripts.course_1_dei

RESULTS_DIR = os.path.join(scripts.course_1_dei.CURRENT_DIR, "Results")

def get_optimal_gp():
    g = gp.GaussianProcess(
        prior_mean_func=gp.mean.Constant(offset=3),
        kernel_func=gp.kernel.Sum(
            gp.kernel.SquaredExponential(
                length_scale=0.06917512071945595,
                kernel_scale=0.029895345214372513,
            ),
            gp.kernel.Periodic(
                period=0.514954586260453,
                length_scale=0.6512752017924203,
                kernel_scale=0.13082827454160073,
            ),
        ),
        noise_std=0.02871806422941413,
    )
    return g
