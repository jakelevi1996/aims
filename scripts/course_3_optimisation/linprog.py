import numpy as np
import cvxpy as cp
import __init__
import util

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

def solve_cp(A, b, c, n):
    x = cp.Variable(n)
    constraints = [A @ x == b, x >= 0]
    objective = cp.Minimize(c @ x)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return problem.value, x.value

rng = np.random.default_rng(2)
util.numpy_set_print_options()

n = 500
m = 100
A, b, c = gen_problem(n, m, rng)

timer = util.Timer()
p, x = solve_cp(A, b, c, n)
timer.print_time_taken()

print(x, p, np.min(x))
