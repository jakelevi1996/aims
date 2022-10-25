# Course 3 - Optimisation

- [Course 3 - Optimisation](#course-3---optimisation)
  - [Installation](#installation)
  - [Solving a toy binary problem with constraints](#solving-a-toy-binary-problem-with-constraints)
  - [Solving a sudoku using `cvxpy`](#solving-a-sudoku-using-cvxpy)

## Installation

Optimisation of convex problems with continuous variables can be performed using the `cvxpy` package. Integer, binary, and mixed problems can be solved by `cvxpy` if the `cvxopt` package is also installed. These packages can be installed as follows:

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

## Solving a sudoku using `cvxpy`

The module [`sudoku.py`](./sudoku_example.py) contains the function `solve` for solving a sudoku. The script [`sudoku_example.py`](./sudoku_example.py) contains an example testing the `sudoku.solve` method:

```python
import time
import sudoku

known_values = [
    [0, 1, 0, 0, 0, 0, 0, 8, 0],
    [8, 0, 4, 0, 0, 0, 7, 0, 6],
    [0, 3, 0, 0, 9, 0, 0, 1, 0],
    [0, 0, 0, 7, 0, 8, 0, 0, 0],
    [0, 0, 5, 0, 1, 0, 8, 0, 0],
    [0, 0, 0, 3, 0, 4, 0, 0, 0],
    [0, 4, 0, 0, 2, 0, 0, 6, 0],
    [5, 0, 2, 0, 0, 0, 3, 0, 9],
    [0, 9, 0, 0, 0, 0, 0, 5, 0],
]
t0 = time.perf_counter()
s = sudoku.solve(known_values)
t1 = time.perf_counter()
print("Solution = \n%s\nSolution found in %.3f seconds" % (s, t1 - t0))
```

This produces the following output:

```
Solution =
[[2. 1. 9. 6. 4. 7. 5. 8. 3.]
 [8. 5. 4. 2. 3. 1. 7. 9. 6.]
 [6. 3. 7. 8. 9. 5. 4. 1. 2.]
 [4. 2. 1. 7. 6. 8. 9. 3. 5.]
 [3. 6. 5. 9. 1. 2. 8. 7. 4.]
 [9. 7. 8. 3. 5. 4. 6. 2. 1.]
 [7. 4. 3. 5. 2. 9. 1. 6. 8.]
 [5. 8. 2. 1. 7. 6. 3. 4. 9.]
 [1. 9. 6. 4. 8. 3. 2. 5. 7.]]
Solution found in 0.541 seconds
```
