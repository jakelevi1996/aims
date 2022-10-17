import gp
import mean
import kernel

def get_optimal_gp():
    g = gp.GaussianProcess(
        prior_mean_func=mean.Constant(offset=3),
        kernel_func=kernel.Sum(
            kernel.SquaredExponential(
                length_scale=0.06917512071945595,
                kernel_scale=0.029895345214372513,
            ),
            kernel.Periodic(
                period=0.514954586260453,
                length_scale=0.6512752017924203,
                kernel_scale=0.13082827454160073,
            ),
        ),
        noise_std=0.02871806422941413,
    )
    return g
