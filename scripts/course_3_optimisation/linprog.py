import numpy as np
import scipy.linalg
import cvxpy as cp
import __init__
import util
import plotting
import scripts.course_3_optimisation

def gen_problem(n=500, m=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    A = rng.normal(size=[m, n])
    x_init = rng.normal(size=n)
    x_init_clipped = np.clip(x_init, 0, None)
    b = A @ x_init_clipped
    c = rng.normal(size=n)
    c[c < 0] *= -1
    return A, b, c

def solve_cp(A, b, c):
    m, n = A.shape
    assert b.size == m
    assert c.size == n
    x = cp.Variable(n)
    constraints = [A @ x == b, x >= 0]
    objective = cp.Minimize(c @ x)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return problem.value, x.value

def solve_pg(
    A,
    b,
    c,
    n_iterations=1000,
    ignore_stationarity=False,
    cache_inverse=False,
):
    m, n = A.shape
    assert b.size == m
    assert c.size == n
    F = np.block(
        [
            [c.reshape(1, n)    , b.reshape(1, m)   , np.zeros([1, n])  ],
            [A                  , np.zeros([m, m])  , np.zeros([m, n])  ],
            [np.zeros([n, n])   , A.T               , -np.identity(n)   ],
        ]
    )
    F_fact = scipy.linalg.lu_factor(F @ F.T)
    g = np.block([0, b, -c])
    z = np.zeros((2 * n) + m)
    clip_inds = np.block([np.ones(n), np.zeros(m), np.ones(n)])
    error_norm_list = []
    for _ in range(n_iterations):
        z_hat = z + F.T @ scipy.linalg.lu_solve(F_fact, g - F @ z)
        z = np.where(clip_inds, np.clip(z_hat, 0, None), z_hat)
        error_norm_list.append(np.linalg.norm(z - z_hat))

    x = z[:n]
    return x, error_norm_list

def solve_dpg(A, b, c, n_iterations=1000, tol=1e-3):
    m, n = A.shape
    assert b.size == m
    assert c.size == n
    F = np.block(
        [
            [c.reshape(1, n)    , b.reshape(1, m)   , np.zeros([1, n])  ],
            [A                  , np.zeros([m, m])  , np.zeros([m, n])  ],
            [np.zeros([n, n])   , A.T               , -np.identity(n)   ],
        ]
    )
    g = np.block([0, b, -c])
    F_proj = F
    g_proj = g
    F_num_rows, F_num_cols = F.shape
    z = np.zeros((2 * n) + m)
    z_inds = np.arange(z.size)
    x_nu_inds = np.block(
        [
            np.full(n, True),
            np.full(m, False),
            np.full(n, True),
        ]
    )

    error_norm_list = []
    for i in range(n_iterations):
        dz = F_proj.T @ np.linalg.solve(
            F_proj @ F_proj.T,
            g_proj - F_proj @ z,
        )
        z += dz
        error_norm_list.append(np.linalg.norm(dz))
        proj_inds = z_inds[np.logical_and(z < -tol, x_nu_inds)]
        if proj_inds.size == 0:
            break

        F_proj = np.block([[F], [np.zeros([proj_inds.size, F_num_cols])]])
        F_proj[F_num_rows + np.arange(proj_inds.size), proj_inds] = 1
        g_proj = np.block([g, np.zeros(proj_inds.size)])

    x = z[:n]
    return x, error_norm_list

rng = np.random.default_rng(2)
util.numpy_set_print_options()

n = 500
m = 100
A, b, c = gen_problem(n, m, rng)

timer = util.Timer()
p, x = solve_cp(A, b, c)
timer.print_time_taken()

timer = util.Timer()
x_pg, error_norm_list = solve_pg(A, b, c)
timer.print_time_taken()

timer = util.Timer()
x_dpg, dpg_error_norm_list = solve_dpg(A, b, c)
timer.print_time_taken()

x_pgp = x_pg + A.T @ np.linalg.solve(A @ A.T, b - A @ x_pg)

negative_inds = np.arange(x_pgp.size)[x_pgp < 0]
m, n = A.shape
A_proj = np.block([[A], [np.zeros([negative_inds.size, n])]])
A_proj[m + np.arange(negative_inds.size), negative_inds] = 1
b_proj = np.block([b, np.zeros(negative_inds.size)])
for i, j in enumerate(negative_inds):
    A_proj[m + i, j] = 1

x_pgdp = x_pgp + A_proj.T @ np.linalg.solve(
    A_proj @ A_proj.T,
    b_proj - A_proj @ x_pgp,
)

print(np.linalg.norm(x - x_pg))
print(np.linalg.norm(x), np.linalg.norm(x_pg))
print(np.max(x), np.max(x_pg))
print(np.min(x), np.min(x_pg))
print(np.linalg.norm(A @ x - b), np.linalg.norm(A @ x_pg - b))
print(np.linalg.norm(c @ x), np.linalg.norm(c @ x_pg))

plotting.plot(
    plotting.Line(np.arange(len(error_norm_list)), error_norm_list, c="b"),
    axis_properties=plotting.AxisProperties(
        xlabel="Iteration",
        log_yscale=True,
        ylim=[0.1, 10],
    ),
    plot_name="$\\Vert z - \\tilde{z} \\Vert_2$",
    dir_name=scripts.course_3_optimisation.RESULTS_DIR,
)
