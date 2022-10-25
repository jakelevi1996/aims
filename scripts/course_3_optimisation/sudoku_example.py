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
