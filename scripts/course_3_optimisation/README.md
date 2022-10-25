# Course 3 - Optimisation

- [Course 3 - Optimisation](#course-3---optimisation)
  - [Installation](#installation)
  - [Solving a toy binary problem with constraints](#solving-a-toy-binary-problem-with-constraints)

## Installation

Optimisation of convex problems with continuous variables can be performed using the `cvxpy` package. Integer, binary, and mixed problems can be solved by `cvxpy` only if the `cvxopt` package is also installed. These packages can be installed as follows:

```
python -m pip install -U pip
python -m pip install cvxpy
python -m pip install cvxopt
```

## Solving a toy binary problem with constraints

Below is an example of solving a toy binary problem using `cvxpy` and `cvxopt`, including a 3D binary variable `x`, and constraints:

```python
import cvxpy as cp

x = cp.Variable(3, boolean=True)
constraints = [
    x[0] + x[1] == 1,
    x[1] + x[2] == 1.
]
objective = cp.Maximize(x[0] + x[1] + x[2])
prob = cp.Problem(objective, constraints)
print(prob.solve())
print(x.value)
```

Output:

```
2.0
[1. 0. 1.]
```
