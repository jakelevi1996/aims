import cvxpy as cp

def solve_separable(x, labels, x_dim, n):
    a = cp.Variable(shape=x_dim)
    b = cp.Variable(shape=1)
    constraints = [cp.multiply(labels, x @ a + b) >= 1]
    objective = cp.Minimize(cp.sum_squares(a))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return a.value, b.value

def solve(x, labels, x_dim, n, norm_penalty=0):
    t = cp.Variable(shape=n)
    a = cp.Variable(shape=x_dim)
    b = cp.Variable(shape=1)
    constraints = [t >= 0, t >= (1 - cp.multiply(labels, x @ a + b))]
    objective = cp.Minimize(cp.sum(t) + norm_penalty * cp.sum_squares(a))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="ECOS")
    return a.value, b.value
